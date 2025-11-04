# Getting Started with Document AI System

This guide will help you set up and run the Document AI System quickly.

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- (Optional) CUDA-compatible GPU for faster processing
- (Windows) Tesseract OCR installed

## Installation

### Step 1: Clone or Download the Project

If you downloaded as ZIP, extract to your desired location.

### Step 2: Create Virtual Environment

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate.bat

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (Deep Learning)
- Transformers (LayoutLMv3)
- FastAPI (REST API)
- spaCy (NLP)
- OpenCV (Image Processing)
- EasyOCR/Tesseract (OCR)
- And more...

### Step 4: Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### Step 5: Install Tesseract OCR (Windows)

1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to default location (C:\Program Files\Tesseract-OCR)
3. Add to PATH or set in code:
   ```python
   import pytesseract
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

## Running the System

### Option 1: Run API Server

```bash
# Using the quick start script
python run_api.py

# Or directly with uvicorn
uvicorn src.api.main:app --reload
```

The API will be available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **API Root**: http://localhost:8000

### Option 2: Use Pipeline Directly (CLI)

```bash
python -m src.pipeline path/to/your/document.pdf
```

This will:
1. Process the document
2. Extract fields
3. Make decisions
4. Generate explainability visualizations
5. Save results to JSON

### Option 3: Use as Python Library

```python
from src.pipeline import DocumentAIPipeline

# Initialize pipeline
pipeline = DocumentAIPipeline()

# Process document
result = pipeline.process_document('invoice.pdf')

print(f"Document Type: {result['document_type']}")
print(f"Decision: {result['decision']}")
print(f"Confidence: {result['confidence_score']:.2%}")
print(f"Fields: {result['fields_extracted']}")
```

## Using the API

### Example: Process a Document

```bash
curl -X POST "http://localhost:8000/process" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@invoice.pdf" \
  -F "generate_explanations=true"
```

Response:
```json
{
  "job_id": "abc-123-def",
  "status": "completed",
  "message": "Document processed successfully"
}
```

### Example: Get Results

```bash
curl -X GET "http://localhost:8000/result/abc-123-def"
```

Response:
```json
{
  "document_type": "invoice",
  "fields_extracted": {
    "invoice_no": "INV-2025-321",
    "total_amount": "₹58,400",
    "vendor": "ABC Solutions Pvt Ltd"
  },
  "decision": "Valid",
  "confidence_score": 0.94,
  "validation": {
    "valid": true,
    "issues": [],
    "warnings": []
  }
}
```

### Example: Rank Resumes

```bash
curl -X POST "http://localhost:8000/rank-resumes" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@resume1.pdf" \
  -F "files=@resume2.pdf" \
  -F "files=@resume3.pdf"
```

## Testing Your Setup

### Quick Test with Sample Data

Create a simple test image:

```python
from PIL import Image, ImageDraw, ImageFont

# Create test invoice image
img = Image.new('RGB', (800, 600), color='white')
d = ImageDraw.Draw(img)

# Add text
d.text((50, 50), "INVOICE", fill='black')
d.text((50, 100), "Invoice #: INV-001", fill='black')
d.text((50, 150), "Vendor: Test Company", fill='black')
d.text((50, 200), "Total: $1,000", fill='black')

img.save('test_invoice.png')
```

Then process it:

```bash
python -m src.pipeline test_invoice.png
```

### Run Unit Tests

```bash
pytest tests/ -v
```

## Configuration

Edit `configs/config.yaml` to customize:

- **Model Settings**: Change model name, max length, etc.
- **OCR Engines**: Enable/disable Tesseract or EasyOCR
- **API Settings**: Change port, file size limits, etc.
- **Reasoning Rules**: Adjust validation thresholds
- **RAG Settings**: Configure vector store and embeddings

Example:

```yaml
model:
  name: "microsoft/layoutlmv3-base"
  num_labels: 3
  max_length: 512

ocr:
  engines:
    - tesseract
    - easyocr
  confidence_threshold: 0.6

api:
  port: 8000
  max_file_size: 10485760  # 10MB
```

## Troubleshooting

### Issue: "No module named 'src'"

**Solution**: Make sure you're in the project root directory and have activated the virtual environment.

### Issue: "Tesseract not found"

**Solution**: 
- Windows: Install Tesseract and add to PATH
- Linux: `sudo apt-get install tesseract-ocr`
- Mac: `brew install tesseract`

### Issue: "CUDA out of memory"

**Solution**: The model will automatically fall back to CPU if GPU memory is insufficient. To force CPU:

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

### Issue: "Model download is slow"

**Solution**: The first run will download LayoutLMv3 (~500MB). Subsequent runs will use cached model.

### Issue: "spaCy model not found"

**Solution**: Run `python -m spacy download en_core_web_sm`

## Next Steps

1. **Prepare Training Data**: Create your own labeled dataset in `data/train/` and `data/val/`
2. **Fine-tune Model**: Use `src/train.py` to train on your data
3. **Deploy to Production**: Containerize with Docker or deploy to cloud
4. **Extend Functionality**: Add new document types or reasoning rules

## Resources

- **API Documentation**: http://localhost:8000/docs (when running)
- **Technical Report**: See `REPORT.md` for detailed architecture
- **Code Documentation**: All modules have inline documentation

## Support

For issues or questions:
1. Check the documentation in `REPORT.md`
2. Review code comments in source files
3. Run tests to verify setup: `pytest tests/`

## Example Workflow

Complete example from start to finish:

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Start API
python run_api.py

# 3. In another terminal, test API
curl http://localhost:8000/health

# 4. Process a document
curl -X POST "http://localhost:8000/process" \
  -F "file=@your_invoice.pdf"

# 5. Check results in outputs/ directory
```
That's it! You're ready to use the Document AI System. 
