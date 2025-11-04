"""
Main Pipeline for Document Understanding System
Orchestrates OCR, NLP, Model Inference, Reasoning, and Explainability
"""
from typing import Dict, Optional
from pathlib import Path
import torch
from loguru import logger
from PIL import Image
import json
import time

from .config import load_config
from .data.document_processor import DocumentProcessor
from .models.multimodal_model import load_model
from .reasoning.decision_engine import ReasoningEngine, RuleBasedReasoning
from .explainability.explainer import ExplainabilityEngine


class DocumentAIPipeline:
    """End-to-end document AI pipeline"""
    
    def __init__(self, config_path: Optional[str] = None, checkpoint_path: Optional[str] = None):
        """
        Initialize the pipeline
        
        Args:
            config_path: Path to configuration file
            checkpoint_path: Path to model checkpoint
        """
        logger.info("Initializing Document AI Pipeline...")
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize document processor
        logger.info("Loading document processor...")
        self.doc_processor = DocumentProcessor(
            ocr_config={
                'engines': self.config.ocr.engines,
                'languages': self.config.ocr.languages
            }
        )
        
        # Load model
        logger.info("Loading AI model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model = load_model(
            config={
                'name': self.config.model.name,
                'num_labels': self.config.model.num_labels,
                'max_length': self.config.model.max_length
            },
            checkpoint_path=checkpoint_path
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize reasoning engine
        logger.info("Loading reasoning engine...")
        self.reasoning_engine = ReasoningEngine(self.config.reasoning.dict())
        self.rule_engine = RuleBasedReasoning()
        
        # Initialize explainability
        logger.info("Loading explainability engine...")
        self.explainer = ExplainabilityEngine(
            model=self.model,
            config={
                **self.config.explainability.dict(),
                'output_dir': self.config.api.output_dir
            }
        )
        
        # Document type mapping
        self.doc_type_map = {
            0: 'invoice',
            1: 'resume',
            2: 'report'
        }
        
        logger.info("Pipeline initialized successfully!")
    
    def process_document(self, file_path: str, generate_explanations: bool = True) -> Dict:
        """
        Process a document end-to-end
        
        Args:
            file_path: Path to document file
            generate_explanations: Whether to generate explainability visualizations
        
        Returns:
            Complete processing result with predictions, extracted fields, and decisions
        """
        start_time = time.time()
        document_id = Path(file_path).stem
        
        logger.info(f"Processing document: {file_path}")
        
        try:
            # Step 1: Document Loading and OCR
            logger.info("Step 1: Document loading and OCR...")
            doc_data = self.doc_processor.process_document(file_path)
            
            text = doc_data['text']
            image = doc_data.get('image')
            entities = doc_data['entities']
            blocks = doc_data.get('blocks', [])
            
            # Extract words and boxes from OCR blocks
            words = [block['text'] for block in blocks] if blocks else text.split()
            boxes = [block['bbox'] for block in blocks] if blocks else []
            
            # If no image (e.g., PDF), create placeholder
            if image is None and blocks:
                # For PDFs, we'd need to convert first page to image
                logger.warning("No image available, using placeholder")
                image = Image.new('RGB', (800, 600), color='white')
            
            # Step 2: Model Inference
            logger.info("Step 2: AI model inference...")
            try:
                prediction = self.model.predict(
                    text=text,
                    image=image,
                    words=words if words else None,
                    boxes=boxes if boxes else None
                )
            except Exception as e:
                logger.error(f"Model prediction failed, falling back to heuristic: {e}")
                prediction = self._heuristic_predict(text)
            
            document_type_id = prediction['document_type']
            document_type = self.doc_type_map.get(document_type_id, 'unknown')
            confidence = prediction['document_type_confidence']
            
            logger.info(f"Predicted document type: {document_type} (confidence: {confidence:.2%})")
            
            # Step 3: Extract domain-specific fields
            logger.info("Step 3: Extracting domain-specific fields...")
            extracted_fields = self._extract_fields_by_type(
                document_type=document_type,
                text=text,
                entities=entities,
                prediction=prediction
            )
            
            # Step 4: Reasoning and Validation
            logger.info("Step 4: Applying reasoning and validation...")
            validation_result = self._validate_document(document_type, extracted_fields)

            # Resume-specific scoring (for single resume processing)
            ranking_info = None
            if document_type == 'resume':
                try:
                    ranking_calc = self.reasoning_engine._calculate_resume_score(extracted_fields, None)
                    ranking_info = {
                        'overall_score': round(ranking_calc.get('total_score', 0) / 100.0, 4),
                        'skills_match': round(ranking_calc.get('breakdown', {}).get('skills_match', 0) / 100.0, 4),
                        'experience_score': round(ranking_calc.get('breakdown', {}).get('experience_years', 0) / 100.0, 4),
                        'key_highlights': [
                            f"Experience: {extracted_fields.get('experience_years', 0)} yrs",
                            f"Education: {', '.join(extracted_fields.get('education', [])) or 'N/A'}",
                            f"Skills: {', '.join(extracted_fields.get('skills', [])[:10])}"
                        ]
                    }
                    # Attach score for decision logic usage
                    extracted_fields['score'] = round(ranking_calc.get('total_score', 0), 2)
                except Exception as _:
                    ranking_info = None
            
            # Apply rule-based reasoning
            rule_violations = self.rule_engine.apply_rules(document_type, extracted_fields)
            
            # Make decision
            decision = self.reasoning_engine.make_decision(
                document_type=document_type,
                extracted_fields=extracted_fields,
                validation_result=validation_result
            )
            
            # Step 5: Generate Explanations
            explanations = {}
            if generate_explanations and image is not None:
                logger.info("Step 5: Generating explainability visualizations...")
                try:
                    explanations = self.explainer.explain(
                        image=image,
                        text=text,
                        words=words if words else text.split()[:50],
                        boxes=boxes if boxes else [[0, 0, 10, 10]] * len(words[:50]),
                        prediction=prediction,
                        document_id=document_id
                    )
                except Exception as e:
                    logger.error(f"Explainability generation failed: {e}")
            
            # Compile results
            processing_time = time.time() - start_time
            
            # Flatten entities for frontend consumption (list of {text,label})
            flat_entities = []
            try:
                for label, items in (entities or {}).items():
                    for it in items:
                        text_val = it.get('text', '') if isinstance(it, dict) else str(it)
                        flat_entities.append({'text': text_val, 'label': label})
            except Exception:
                flat_entities = []

            # Ensure explainability_map has a sensible default path if generated file exists
            summary_map_path = explanations.get('summary', '')
            if not summary_map_path:
                candidate = Path(self.config.api.output_dir) / f"{document_id}_summary.png"
                if candidate.exists():
                    summary_map_path = str(candidate)

            # For invoices, ensure required keys exist in fields_extracted
            if document_type == 'invoice':
                extracted_fields.setdefault('invoice_no', extracted_fields.get('invoice_no', ''))
                extracted_fields.setdefault('total_amount', extracted_fields.get('total_amount', ''))
                extracted_fields.setdefault('vendor', extracted_fields.get('vendor', ''))

            # Optionally round confidence for API consumers (keep numeric 0-1 scale)
            try:
                rounded_confidence = round(float(confidence), 2)
            except Exception:
                rounded_confidence = confidence

            result = {
                'document_type': document_type,
                'fields_extracted': extracted_fields,
                'decision': decision['decision'],
                'confidence_score': rounded_confidence,
                'validation': validation_result,
                'rule_violations': rule_violations,
                # Frontend-friendly additions
                'extracted_text': text,
                'entities': flat_entities,
                'predictions': {'confidence': confidence},
                # Keep explainability artifacts
                'explainability_map': summary_map_path,
                'attention_map': explanations.get('attention_map', ''),
                'token_importance': explanations.get('token_importance', ''),
                'processing_time_seconds': round(processing_time, 2),
                'metadata': {
                    'file_name': Path(file_path).name,
                    'document_id': document_id,
                    'model_name': self.config.model.name
                },
                # Structured reasoning object for UI tabs
                'reasoning': {
                    'validation': validation_result if document_type == 'invoice' else None,
                    'ranking': ranking_info if document_type == 'resume' else None,
                    'insights': decision.get('reasoning', [])
                }
            }
            
            logger.info(f"Document processed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    
    def _extract_fields_by_type(
        self,
        document_type: str,
        text: str,
        entities: Dict,
        prediction: Dict
    ) -> Dict:
        """Extract fields based on document type"""
        
        if document_type == 'invoice':
            fields = self.doc_processor.extract_invoice_fields(text, entities)
        elif document_type == 'resume':
            fields = self.doc_processor.extract_resume_fields(text, entities)
        elif document_type == 'report':
            fields = self.doc_processor.extract_report_fields(text, entities)
        else:
            fields = {}
        
        # Add entities from model prediction if available
        if 'token_predictions' in prediction:
            model_entities = self.model.extract_entities(
                words=text.split()[:len(prediction['token_predictions'])],
                token_predictions=prediction['token_predictions']
            )
            fields['model_entities'] = model_entities
        
        return fields
    
    def _validate_document(self, document_type: str, fields: Dict) -> Dict:
        """Validate document based on type"""
        
        if document_type == 'invoice':
            return self.reasoning_engine.validate_invoice(fields)
        elif document_type == 'resume':
            # For resume, validation is implicit in scoring
            return {
                'valid': True,
                'issues': [],
                'warnings': [],
                'validation_details': {}
            }
        elif document_type == 'report':
            return self.reasoning_engine.validate_report(fields)
        else:
            return {
                'valid': False,
                'issues': ['Unknown document type'],
                'warnings': [],
                'validation_details': {}
            }
    
    def _heuristic_predict(self, text: str) -> Dict:
        """Simple heuristic prediction when model fails"""
        t = (text or "").lower()
        if any(k in t for k in ["invoice", "amount due", "subtotal", "bill to", "tax"]):
            doc_id = 0
        elif any(k in t for k in ["resume", "curriculum vitae", "skills", "experience"]):
            doc_id = 1
        else:
            doc_id = 2
        return {
            'document_type': doc_id,
            'document_type_confidence': 0.6,
            'token_predictions': [],
            'reasoning_features': [0.0]*128,
            'attentions': []
        }

    def batch_process(self, file_paths: list[str], generate_explanations: bool = False) -> list[Dict]:
        """Process multiple documents in batch"""
        results = []
        
        for file_path in file_paths:
            try:
                result = self.process_document(file_path, generate_explanations)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results.append({
                    'error': str(e),
                    'file_path': file_path
                })
        
        return results
    
    def rank_resumes(self, resume_files: list[str], job_requirements: Optional[Dict] = None) -> list[Dict]:
        """Process and rank multiple resumes"""
        logger.info(f"Ranking {len(resume_files)} resumes...")
        
        # Process all resumes
        resumes_data = []
        for file_path in resume_files:
            try:
                result = self.process_document(file_path, generate_explanations=False)
                if result['document_type'] == 'resume':
                    resumes_data.append({
                        **result['fields_extracted'],
                        'file_path': file_path,
                        'confidence': result['confidence_score']
                    })
            except Exception as e:
                logger.error(f"Failed to process resume {file_path}: {e}")
        
        # Rank resumes
        ranked = self.reasoning_engine.rank_resumes(resumes_data, job_requirements)
        
        logger.info("Resume ranking completed")
        return ranked
    
    def save_result(self, result: Dict, output_path: str):
        """Save processing result to JSON"""
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Result saved to: {output_path}")


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline <document_path>")
        sys.exit(1)
    
    document_path = sys.argv[1]
    
    # Initialize pipeline
    pipeline = DocumentAIPipeline()
    
    # Process document
    result = pipeline.process_document(document_path)
    
    # Print result
    print(json.dumps(result, indent=2))
    
    # Save result
    output_path = f"{Path(document_path).stem}_result.json"
    pipeline.save_result(result, output_path)


if __name__ == "__main__":
    main()
