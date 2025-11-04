"""
Multi-Modal Transformer Model for Document Understanding
Uses LayoutLMv3 for visual + textual feature fusion
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from loguru import logger
import numpy as np
from PIL import Image

# Attempt to import transformers; fall back if unavailable
try:
    from transformers import (
        LayoutLMv3Processor,
        LayoutLMv3ForTokenClassification,
        LayoutLMv3ForSequenceClassification,
        LayoutLMv3Model
    )
    _TRANS_AVAILABLE = True
except Exception as _e:
    logger.warning(f"Transformers not available or failed to import: {_e}")
    _TRANS_AVAILABLE = False


class DocumentUnderstandingModel(nn.Module):
    """
    Multi-task model for document classification and entity extraction
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.model_name = config.get('name', 'microsoft/layoutlmv3-base')
        self.num_labels = config.get('num_labels', 3)
        self.max_length = config.get('max_length', 512)
        
        # Load pre-trained LayoutLMv3
        if not _TRANS_AVAILABLE:
            raise RuntimeError("Transformers not available for DocumentUnderstandingModel")
        logger.info(f"Loading model: {self.model_name}")
        self.backbone = LayoutLMv3Model.from_pretrained(self.model_name)
        self.processor = LayoutLMv3Processor.from_pretrained(
            self.model_name, 
            apply_ocr=False
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.backbone.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.num_labels)
        )
        
        # Entity extraction head (for token classification)
        self.entity_labels = {
            'O': 0,  # Outside
            'B-INVOICE_NO': 1,
            'B-AMOUNT': 2,
            'B-DATE': 3,
            'B-VENDOR': 4,
            'B-NAME': 5,
            'B-EMAIL': 6,
            'B-PHONE': 7,
            'B-SKILL': 8,
            'B-EDUCATION': 9,
            'I-INVOICE_NO': 10,
            'I-AMOUNT': 11,
            'I-DATE': 12,
            'I-VENDOR': 13,
            'I-NAME': 14,
        }
        
        self.token_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.backbone.config.hidden_size, len(self.entity_labels))
        )
        
        # Reasoning module
        self.reasoning_layer = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bbox: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        token_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            bbox: Bounding boxes [batch_size, seq_len, 4]
            pixel_values: Image pixels [batch_size, 3, H, W]
            labels: Document type labels [batch_size]
            token_labels: Token-level labels [batch_size, seq_len]
        """
        # Get backbone outputs
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            output_attentions=True,
            output_hidden_states=True
        )
        
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]
        pooled_output = outputs.pooler_output  # [batch, hidden]
        
        # Document classification
        class_logits = self.classifier(pooled_output)
        
        # Token classification (entity extraction)
        token_logits = self.token_classifier(sequence_output)
        
        # Reasoning features
        reasoning_features = self.reasoning_layer(pooled_output)
        
        # Calculate losses if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            class_loss = loss_fct(class_logits, labels)
            
            if token_labels is not None:
                token_loss = loss_fct(
                    token_logits.view(-1, len(self.entity_labels)),
                    token_labels.view(-1)
                )
                loss = class_loss + token_loss
            else:
                loss = class_loss
        
        return {
            'loss': loss,
            'class_logits': class_logits,
            'token_logits': token_logits,
            'reasoning_features': reasoning_features,
            'attentions': outputs.attentions,
            'hidden_states': outputs.hidden_states
        }
    
    def prepare_inputs(
        self, 
        text: str, 
        image: Image.Image,
        words: Optional[List[str]] = None,
        boxes: Optional[List[List[int]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the model
        
        Args:
            text: Input text
            image: Input image
            words: List of words (from OCR)
            boxes: List of bounding boxes for words
        """
        if words is None or boxes is None:
            # Use processor's OCR
            encoding = self.processor(
                image, 
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
        else:
            # Use provided OCR results
            encoding = self.processor(
                image,
                text=words,
                boxes=boxes,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                stride=128,
                return_overflowing_tokens=True
            )
        
        return encoding
    
    def predict(
        self, 
        text: str, 
        image: Image.Image,
        words: Optional[List[str]] = None,
        boxes: Optional[List[List[int]]] = None
    ) -> Dict:
        """
        Make predictions on a document
        """
        self.eval()
        
        # Convert empty lists to None
        if words is not None and len(words) == 0:
            words = None
        if boxes is not None and len(boxes) == 0:
            boxes = None
        
        with torch.no_grad():
            # Prepare inputs
            inputs = self.prepare_inputs(text, image, words, boxes)
            
            # Move to device
            device = next(self.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = self(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                bbox=inputs.get('bbox', torch.zeros((1, self.max_length, 4), device=device)),
                pixel_values=inputs['pixel_values']
            )
            
            # Get predictions
            class_probs = torch.softmax(outputs['class_logits'], dim=-1)
            class_pred = torch.argmax(class_probs, dim=-1)
            
            token_probs = torch.softmax(outputs['token_logits'], dim=-1)
            token_preds = torch.argmax(token_probs, dim=-1)
            
            # Handle attentions safely
            attentions_list = []
            if outputs['attentions'] is not None:
                try:
                    attentions_list = [att[0].cpu().numpy() if hasattr(att, 'cpu') else att for att in outputs['attentions']]
                except:
                    attentions_list = []
            
            return {
                'document_type': class_pred.item(),
                'document_type_confidence': class_probs[0, class_pred].item(),
                'token_predictions': token_preds[0].cpu().numpy(),
                'reasoning_features': outputs['reasoning_features'][0].cpu().numpy(),
                'attentions': attentions_list
            }
    
    def extract_entities(
        self, 
        words: List[str], 
        token_predictions: np.ndarray
    ) -> List[Dict]:
        """
        Extract entities from token predictions
        """
        entities = []
        current_entity = None
        
        id_to_label = {v: k for k, v in self.entity_labels.items()}
        
        for word, pred_id in zip(words, token_predictions):
            label = id_to_label.get(pred_id, 'O')
            
            if label.startswith('B-'):
                # Start of new entity
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label[2:]
                current_entity = {
                    'type': entity_type,
                    'text': word,
                    'confidence': 1.0
                }
            elif label.startswith('I-') and current_entity:
                # Continuation of entity
                current_entity['text'] += ' ' + word
            elif label == 'O' and current_entity:
                # End of entity
                entities.append(current_entity)
                current_entity = None
        
        # Add last entity if exists
        if current_entity:
            entities.append(current_entity)
        
        return entities


class DocumentClassifier(nn.Module):
    """Specialized document classifier"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
            config.get('name', 'microsoft/layoutlmv3-base'),
            num_labels=config.get('num_labels', 3)
        )
        self.processor = LayoutLMv3Processor.from_pretrained(
            config.get('name', 'microsoft/layoutlmv3-base'),
            apply_ocr=False
        )
    
    def forward(self, **inputs):
        return self.model(**inputs)
    
    def predict(self, image: Image.Image, words: List[str], boxes: List[List[int]]) -> Dict:
        self.eval()
        
        with torch.no_grad():
            encoding = self.processor(
                image,
                text=words,
                boxes=boxes,
                return_tensors="pt",
                padding="max_length",
                truncation=True
            )
            
            device = next(self.parameters()).device
            encoding = {k: v.to(device) for k, v in encoding.items()}
            
            outputs = self.model(**encoding)
            
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)
            
            return {
                'document_type': pred.item(),
                'confidence': probs[0, pred].item(),
                'all_probabilities': probs[0].cpu().numpy()
            }


class SimpleHeuristicModel:
    """Lightweight fallback when transformers weights are unavailable.
    Provides the minimal interface used by the pipeline (to, eval, predict, extract_entities).
    """
    def __init__(self, config: Dict):
        self.max_length = config.get('max_length', 512)

    def to(self, device):
        return self

    def eval(self):
        return self

    def predict(self, text: str, image: Image.Image, words: Optional[List[str]] = None, boxes: Optional[List[List[int]]] = None) -> Dict:
        t = (text or "").lower()
        # Naive heuristics
        if any(k in t for k in ["invoice", "amount due", "subtotal", "bill to"]):
            doc_id = 0
        elif any(k in t for k in ["resume", "curriculum vitae", "skills", "experience"]):
            doc_id = 1
        else:
            doc_id = 2
        confidence = 0.75
        length = min(self.max_length, len(words) if words else 128)
        token_predictions = np.zeros((length,), dtype=np.int64)
        reasoning_features = np.zeros((128,), dtype=np.float32)
        return {
            'document_type': doc_id,
            'document_type_confidence': confidence,
            'token_predictions': token_predictions,
            'reasoning_features': reasoning_features,
            'attentions': []
        }

    def extract_entities(self, words: List[str], token_predictions: np.ndarray) -> List[Dict]:
        return []


def load_model(config: Dict, checkpoint_path: Optional[str] = None) -> DocumentUnderstandingModel:
    """Load model from checkpoint or create new; fall back to heuristic if unavailable"""
    try:
        model = DocumentUnderstandingModel(config)
        if checkpoint_path:
            logger.info(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
        return model
    except Exception as e:
        logger.warning(f"Falling back to SimpleHeuristicModel due to error loading model: {e}")
        return SimpleHeuristicModel(config)
