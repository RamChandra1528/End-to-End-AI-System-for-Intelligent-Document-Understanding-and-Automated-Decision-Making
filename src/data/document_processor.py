"""
Document Processor with NLP pipeline for text preprocessing and normalization
"""
import re
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
from PIL import Image
from loguru import logger
from .ocr_engine import OCREngine

# Optional spaCy import
try:
    import spacy  # type: ignore
except Exception as _e:
    spacy = None  # type: ignore
    logger.warning(f"spaCy not available: {_e}")


class DocumentProcessor:
    """Process documents with OCR and NLP preprocessing"""
    
    def __init__(self, ocr_config: Dict, nlp_model: str = "en_core_web_sm"):
        self.ocr_engine = OCREngine(
            engines=ocr_config.get('engines', ['tesseract']),
            languages=ocr_config.get('languages', ['en'])
        )
        
        # Load spaCy model
        try:
            if spacy is None:
                raise OSError("spaCy not installed")
            self.nlp = spacy.load(nlp_model)
        except OSError:
            logger.warning(f"spaCy model {nlp_model} not found or spaCy not installed. Download it with: python -m spacy download {nlp_model}")
            self.nlp = None
    
    def load_document(self, file_path: str) -> Dict:
        """Load document and extract initial information"""
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        
        doc_info = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_type': file_ext
        }
        
        # Extract based on file type
        if file_ext == '.pdf':
            pages = self.ocr_engine.extract_from_pdf(str(file_path))
            doc_info['pages'] = pages
            doc_info['text'] = ' '.join([p['text'] for p in pages])
            doc_info['image'] = None
        elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            ocr_result = self.ocr_engine.extract_text(str(file_path))
            doc_info['text'] = ocr_result['text']
            doc_info['blocks'] = ocr_result['blocks']
            doc_info['ocr_engine'] = ocr_result['engine']
            
            # Load image for visual features
            doc_info['image'] = Image.open(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        return doc_info
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\:\-\$\%\@\#\/\\]', '', text)
        
        # Normalize currency symbols
        text = text.replace('₹', 'INR ')
        text = text.replace('$', 'USD ')
        text = text.replace('€', 'EUR ')
        
        return text.strip()
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict]]:
        """Extract named entities using spaCy"""
        if self.nlp is None:
            return {'entities': []}
        
        doc = self.nlp(text)
        
        entities = {
            'PERSON': [],
            'ORG': [],
            'DATE': [],
            'MONEY': [],
            'GPE': [],  # Geo-political entity
            'CARDINAL': [],  # Numbers
            'EMAIL': [],
            'PHONE': [],
            'OTHER': []
        }
        
        # Extract spaCy entities
        for ent in doc.ents:
            entity_info = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            }
            
            if ent.label_ in entities:
                entities[ent.label_].append(entity_info)
            else:
                entities['OTHER'].append(entity_info)
        
        # Extract emails using regex
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            entities['EMAIL'].append({
                'text': match.group(),
                'label': 'EMAIL',
                'start': match.start(),
                'end': match.end()
            })
        
        # Extract phone numbers
        phone_pattern = r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
        for match in re.finditer(phone_pattern, text):
            entities['PHONE'].append({
                'text': match.group(),
                'label': 'PHONE',
                'start': match.start(),
                'end': match.end()
            })
        
        return entities
    
    def extract_invoice_fields(self, text: str, entities: Dict) -> Dict:
        """Extract invoice-specific fields"""
        fields = {}
        
        # Extract invoice number
        inv_pattern = r'(?:invoice|inv)[\s#:]*([A-Z0-9\-]+)'
        inv_match = re.search(inv_pattern, text, re.IGNORECASE)
        if inv_match:
            fields['invoice_no'] = inv_match.group(1)
        
        # Extract vendor/company name (usually near top)
        if entities.get('ORG'):
            fields['vendor'] = entities['ORG'][0]['text']
        
        # Extract date
        if entities.get('DATE'):
            fields['date'] = entities['DATE'][0]['text']
        
        # Extract total amount
        total_pattern = r'(?:total|amount due|balance)[\s:]*(?:INR|USD|EUR|\$|₹)?\s*([\d,]+\.?\d*)'
        total_match = re.search(total_pattern, text, re.IGNORECASE)
        if total_match:
            fields['total_amount'] = total_match.group(1).replace(',', '')
        elif entities.get('MONEY'):
            # Use last money entity as total
            fields['total_amount'] = entities['MONEY'][-1]['text']
        
        # Extract line items
        fields['line_items'] = self._extract_line_items(text)
        
        return fields
    
    def extract_resume_fields(self, text: str, entities: Dict) -> Dict:
        """Extract resume-specific fields"""
        fields = {}
        
        # Extract name (usually first PERSON entity)
        if entities.get('PERSON'):
            fields['name'] = entities['PERSON'][0]['text']
        
        # Extract email
        if entities.get('EMAIL'):
            fields['email'] = entities['EMAIL'][0]['text']
        
        # Extract phone
        if entities.get('PHONE'):
            fields['phone'] = entities['PHONE'][0]['text']
        
        # Extract experience (years)
        exp_pattern = r'(\d+)[\+]?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)'
        exp_match = re.search(exp_pattern, text, re.IGNORECASE)
        if exp_match:
            fields['experience_years'] = int(exp_match.group(1))
        
        # Extract education
        edu_keywords = ['bachelor', 'master', 'phd', 'mba', 'b.tech', 'm.tech', 'b.sc', 'm.sc']
        education = []
        for keyword in edu_keywords:
            if keyword in text.lower():
                education.append(keyword.upper())
        fields['education'] = list(set(education))
        
        # Extract skills
        fields['skills'] = self._extract_skills(text)
        
        return fields
    
    def extract_report_fields(self, text: str, entities: Dict) -> Dict:
        """Extract report-specific fields"""
        fields = {}
        
        # Extract title (usually first line or bold text)
        lines = text.split('\n')
        if lines:
            fields['title'] = lines[0].strip()
        
        # Extract sections
        section_pattern = r'(?:^|\n)((?:Abstract|Introduction|Methodology|Results|Conclusion|References)[:\s]*)'
        sections = re.finditer(section_pattern, text, re.IGNORECASE)
        fields['sections'] = [s.group(1).strip(':').strip() for s in sections]
        
        # Word count
        fields['word_count'] = len(text.split())
        
        # Extract key dates
        if entities.get('DATE'):
            fields['dates'] = [d['text'] for d in entities['DATE']]
        
        return fields
    
    def _extract_line_items(self, text: str) -> List[Dict]:
        """Extract line items from invoice"""
        line_items = []
        
        # Pattern: Description ... Quantity ... Price
        item_pattern = r'([A-Za-z\s]+)\s+(\d+)\s+(?:INR|USD|\$|₹)?\s*([\d,]+\.?\d*)'
        
        for match in re.finditer(item_pattern, text):
            line_items.append({
                'description': match.group(1).strip(),
                'quantity': int(match.group(2)),
                'price': float(match.group(3).replace(',', ''))
            })
        
        return line_items
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract technical skills from resume"""
        # Common skills database
        common_skills = [
            'python', 'java', 'javascript', 'c++', 'react', 'node.js', 'angular',
            'machine learning', 'deep learning', 'nlp', 'computer vision',
            'tensorflow', 'pytorch', 'keras', 'sql', 'mongodb', 'aws', 'azure',
            'docker', 'kubernetes', 'git', 'agile', 'scrum'
        ]
        
        text_lower = text.lower()
        found_skills = [skill for skill in common_skills if skill in text_lower]
        
        return found_skills
    
    def process_document(self, file_path: str) -> Dict:
        """Complete document processing pipeline"""
        logger.info(f"Processing document: {file_path}")
        
        # Load document
        doc_info = self.load_document(file_path)
        
        # Preprocess text
        clean_text = self.preprocess_text(doc_info['text'])
        doc_info['clean_text'] = clean_text
        
        # Extract entities
        entities = self.extract_entities(clean_text)
        doc_info['entities'] = entities
        
        # Store for return
        result = {
            'file_info': {
                'path': doc_info['file_path'],
                'name': doc_info['file_name'],
                'type': doc_info['file_type']
            },
            'text': clean_text,
            'raw_text': doc_info['text'],
            'entities': entities,
            'image': doc_info.get('image'),
            'blocks': doc_info.get('blocks', [])
        }
        
        return result
