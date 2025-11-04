"""
OCR Engine Module supporting multiple OCR backends
"""
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import pytesseract
from pathlib import Path
from loguru import logger

# Optional dependencies
try:
    import easyocr  # type: ignore
except Exception as _e:
    easyocr = None  # type: ignore
    logger.warning(f"easyocr not available: {_e}")

try:
    import pdf2image  # type: ignore
except Exception as _e:
    pdf2image = None  # type: ignore
    logger.warning(f"pdf2image not available: {_e}")


class OCREngine:
    """Multi-engine OCR processor"""
    
    def __init__(self, engines: List[str] = ["tesseract"], languages: List[str] = ["en"]):
        self.engines = engines
        self.languages = languages
        self.easyocr_reader = None
        
        if "easyocr" in engines:
            if easyocr is None:
                logger.warning("EasyOCR requested but not installed. Disabling easyocr engine.")
                self.engines = [e for e in engines if e != "easyocr"]
            else:
                logger.info("Initializing EasyOCR...")
                try:
                    self.easyocr_reader = easyocr.Reader(languages, gpu=False)  # Use CPU
                except Exception as e:
                    logger.warning(f"EasyOCR initialization failed: {e}. Will use Tesseract only.")
                    self.engines = [e for e in engines if e != "easyocr"]
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR accuracy"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Deskew if needed
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
            if abs(angle) > 0.5:  # Only deskew if angle is significant
                (h, w) = binary.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                binary = cv2.warpAffine(
                    binary, M, (w, h),
                    flags=cv2.INTER_CUBIC, 
                    borderMode=cv2.BORDER_REPLICATE
                )
        
        return binary
    
    def extract_with_tesseract(self, image: np.ndarray) -> Dict:
        """Extract text using Tesseract OCR"""
        try:
            # Get detailed OCR data
            data = pytesseract.image_to_data(
                image, 
                output_type=pytesseract.Output.DICT,
                lang='+'.join(self.languages)
            )
        except Exception as tesseract_error:
            # If tesseract is not installed, return mock data
            logger.warning(f"Tesseract not available: {tesseract_error}. Using mock OCR.")
            return {
                'text': 'DEMO MODE - Tesseract not installed. Please install Tesseract OCR for full functionality.',
                'blocks': [{
                    'text': 'Demo',
                    'confidence': 1.0,
                    'bbox': [10, 10, 100, 50]
                }],
                'engine': 'mock'
            }
        
        try:
            
            # Extract text and bounding boxes
            text_blocks = []
            full_text = []
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:  # Filter out low confidence
                    text = data['text'][i].strip()
                    if text:
                        text_blocks.append({
                            'text': text,
                            'confidence': float(data['conf'][i]) / 100.0,
                            'bbox': [
                                data['left'][i],
                                data['top'][i],
                                data['left'][i] + data['width'][i],
                                data['top'][i] + data['height'][i]
                            ]
                        })
                        full_text.append(text)
            
            return {
                'text': ' '.join(full_text),
                'blocks': text_blocks,
                'engine': 'tesseract'
            }
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return {'text': '', 'blocks': [], 'engine': 'tesseract'}
    
    def extract_with_easyocr(self, image: np.ndarray) -> Dict:
        """Extract text using EasyOCR"""
        try:
            if easyocr is None:
                raise RuntimeError("easyocr not available")
            if self.easyocr_reader is None:
                self.easyocr_reader = easyocr.Reader(self.languages, gpu=False)
            
            results = self.easyocr_reader.readtext(image)
            
            text_blocks = []
            full_text = []
            
            for bbox, text, confidence in results:
                # Convert bbox format
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                text_blocks.append({
                    'text': text,
                    'confidence': float(confidence),
                    'bbox': [
                        min(x_coords), 
                        min(y_coords),
                        max(x_coords), 
                        max(y_coords)
                    ]
                })
                full_text.append(text)
            
            return {
                'text': ' '.join(full_text),
                'blocks': text_blocks,
                'engine': 'easyocr'
            }
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return {'text': '', 'blocks': [], 'engine': 'easyocr'}
    
    def extract_text(self, image_path: str, preprocess: bool = True) -> Dict:
        """Extract text from image using configured engines"""
        # Load image
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
            if image is None:
                # Try with PIL
                pil_image = Image.open(image_path)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        else:
            image = image_path
        
        # Preprocess if requested
        if preprocess:
            processed_image = self.preprocess_image(image)
        else:
            processed_image = image
        
        # Run all configured engines
        results = {}
        
        if "tesseract" in self.engines:
            results['tesseract'] = self.extract_with_tesseract(processed_image)
        
        if "easyocr" in self.engines:
            results['easyocr'] = self.extract_with_easyocr(processed_image)
        
        # Merge results (prefer higher confidence)
        return self.merge_results(results)
    
    def merge_results(self, results: Dict) -> Dict:
        """Merge results from multiple OCR engines"""
        if len(results) == 1:
            return list(results.values())[0]
        
        # For now, prefer EasyOCR if available, else Tesseract
        if 'easyocr' in results and results['easyocr']['text']:
            return results['easyocr']
        elif 'tesseract' in results and results['tesseract']['text']:
            return results['tesseract']
        else:
            return {'text': '', 'blocks': [], 'engine': 'none'}
    
    def extract_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text from PDF pages"""
        try:
            if pdf2image is None:
                logger.warning("pdf2image not available; cannot process PDF")
                return []
            # Convert PDF to images
            images = pdf2image.convert_from_path(pdf_path)
            
            results = []
            for i, image in enumerate(images):
                # Convert PIL to OpenCV
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Extract text
                page_result = self.extract_text(cv_image)
                page_result['page'] = i + 1
                results.append(page_result)
            
            return results
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return []


def test_ocr():
    """Test OCR functionality"""
    ocr = OCREngine(engines=["tesseract", "easyocr"])
    
    # Create a test image with text
    test_image = np.ones((200, 500, 3), dtype=np.uint8) * 255
    cv2.putText(test_image, "Invoice #12345", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    result = ocr.extract_text(test_image)
    print("OCR Result:", result)


if __name__ == "__main__":
    test_ocr()
