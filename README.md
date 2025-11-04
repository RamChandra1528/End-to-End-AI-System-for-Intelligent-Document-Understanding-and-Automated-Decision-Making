# Document AI System

An end-to-end AI system for intelligent document understanding and automated decision-making. Supports OCR, multi-modal deep learning (LayoutLMv3), explainability (attention maps, SHAP), rule-based reasoning, RAG (optional), and a FastAPI REST API.

## Features
- Document types: invoice, resume, report
- OCR: Tesseract + EasyOCR, PDF support
- NLP preprocessing: spaCy-based cleaning, entity extraction (regex + NER)
- Multi-modal transformer: LayoutLMv3 for visual+text fusion
- Tasks: document classification, entity extraction, reasoning
- Explainability: attention heatmaps, token importance, SHAP (gradient-based)
- Reasoning layer: invoice validation, resume ranking, report checks, rules engine
- REST API: FastAPI endpoints for upload, batch, ranking, validation
- Optional RAG + Knowledge Graph

## Project Structure
```
document_ai_system/
├─ src/
│  ├─ api/
│  │  └─ main.py
│  ├─ data/
│  │  ├─ ocr_engine.py
│  │  └─ document_processor.py
│  ├─ explainability/
│  │  └─ explainer.py
│  ├─ models/
│  │  └─ multimodal_model.py
│  ├─ rag/
│  │  └─ rag_engine.py
│  ├─ reasoning/
│  │  └─ decision_engine.py
│  ├─ config.py
│  ├─ pipeline.py
│  └─ train.py
├─ configs/
│  └─ config.yaml
├─ outputs/
├─ checkpoints/
├─ tests/
├─ requirements.txt
└─ README.md
```

## Setup
1. Create a Python 3.10+ virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```
4. (Windows) Install Tesseract OCR:
   - Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Add tesseract.exe to PATH or set pytesseract.pytesseract.tesseract_cmd

## Configuration
Edit `configs/config.yaml` for model, OCR, API, reasoning, and RAG settings.

## Run API
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Open http://localhost:8000/docs for Swagger UI.

## Example Usage (Pipeline CLI)
```bash
python -m src.pipeline path/to/document.pdf
```

## Training
- Prepare `data/train/manifest.json` and `data/val/manifest.json` with entries like:
```json
{
  "image": "invoices/inv1.png",
  "label": 0,
  "words": ["Invoice", "#", "INV-2025-321", ...],
  "boxes": [[x1,y1,x2,y2], ...],
  "token_labels": [0,0,1,10,...]
}
```
- Start training by customizing `src/train.py` and uncommenting dataset/trainer lines.

## Output JSON example
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
  "explainability_map": "outputs/123_summary.png"
}
```

## Notes
- LayoutLMv3 requires images and word bounding boxes; when missing, the processor falls back to internal OCR assumptions.
- SHAP is approximated via gradients for efficiency.
- RAG/Knowledge Graph modules are optional and will be no-ops if dependencies are missing.

## Testing
```bash
pytest
```

## Deployment
- Containerize with Uvicorn/Gunicorn.
- Optional: deploy to Hugging Face Spaces, Render, or AWS Lambda (with serverless frameworks).
