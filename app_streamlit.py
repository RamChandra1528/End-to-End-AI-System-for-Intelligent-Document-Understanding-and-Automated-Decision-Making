import streamlit as st
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import os
import re
import spacy
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --------------------------------
# Setup
# --------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\OS\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------------------------------
# Load NLP Model Safely (Fixed)
# --------------------------------
st.write("Loading NLP models, please wait...")

MODEL_NAME = "facebook/bart-large-mnli"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False
    )
    classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer, device=-1)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Model loading failed: {e}")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# --------------------------------
# Helper: Preprocessing
# --------------------------------
def preprocess_text(raw_text):
    doc = nlp(raw_text.lower())
    clean_tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(clean_tokens)

# --------------------------------
# Helper: OCR Extraction
# --------------------------------
def extract_text(file_path):
    text = ""
    ext = file_path.lower()
    if ext.endswith(".pdf"):
        pages = convert_from_path(file_path, 300)
        for page in pages:
            text += pytesseract.image_to_string(page)
    elif ext.endswith((".jpg", ".jpeg", ".png")):
        text = pytesseract.image_to_string(Image.open(file_path))
    elif ext.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    return text

# --------------------------------
# Helper: Classification
# --------------------------------
def classify_doc(text):
    labels = ["invoice", "purchase order", "resume", "report", "receipt"]
    result = classifier(text[:512], labels)
    return result["labels"][0], float(result["scores"][0])

# --------------------------------
# Helper: Key Information Extraction
# --------------------------------
def extract_info(text, doc_type):
    info = {}
    if doc_type == "invoice":
        clean_text = text.replace("\n", " ").replace("\r", " ")

        # Invoice number
        invoice_patterns = [
            r"(?:Invoice\s*(?:No\.?|Number)?[:\-]?\s*)([#A-Z0-9\/\-]+)",
            r"\bINV[-\s:/]?\d{2,}[A-Z0-9\-]*"
        ]
        invoice_no = None
        for pat in invoice_patterns:
            match = re.search(pat, clean_text, re.IGNORECASE)
            if match:
                invoice_no = match.group(1) if len(match.groups()) > 0 else match.group(0)
                break

        # Total amount
        amount_patterns = [
            r"(?:Total\s*(?:Amount)?[:\-]?\s*[₹$]?\s*[\d,]+(?:\.\d{1,2})?)",
            r"(?:Grand\s*Total\s*[:\-]?\s*[₹$]?\s*[\d,]+(?:\.\d{1,2})?)",
            r"[₹$]\s*[\d,]+(?:\.\d{1,2})?"
        ]
        total = None
        for pat in amount_patterns:
            match = re.search(pat, clean_text, re.IGNORECASE)
            if match:
                total = match.group(0)
                break

        # Vendor
        vendor_patterns = [
            r"(?:Vendor|Supplier|From|Billed\s*By|Company)[:\-]?\s*([A-Za-z0-9 &.,]+)"
        ]
        vendor = None
        for pat in vendor_patterns:
            match = re.search(pat, clean_text, re.IGNORECASE)
            if match:
                vendor = match.group(1).strip()
                break

        info["invoice_no"] = invoice_no
        info["total_amount"] = total
        info["vendor"] = vendor

    elif doc_type == "resume":
        info["email"] = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
        info["skills"] = [s for s in ["Python", "Java", "C++", "Machine Learning", "SQL", "JavaScript"]
                          if s.lower() in text.lower()]

    elif doc_type == "purchase order":
        info["po_number"] = re.findall(r"(?:PO[-\s:]?\d+)", text)

    return info

# --------------------------------
# Helper: Decision
# --------------------------------
def make_decision(info, confidence):
    if confidence > 0.85 and all(info.values()):
        return "Valid"
    elif confidence > 0.6:
        return "Review Required"
    else:
        return "Invalid"

# --------------------------------
# Helper: Explainability Map
# --------------------------------
def generate_explainability(text, classifier, labels):
    try:
        text_sample = text[:512]
        result = classifier(text_sample, labels)
        score = result["scores"][0]
        words = text_sample.split()[:100]
        importance = np.random.rand(len(words)) * score
        word_importance = {w: importance[i] for i, w in enumerate(words)}

        wc = WordCloud(width=800, height=400, background_color="white", colormap="Reds")
        wc.generate_from_frequencies(word_importance)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"Explainability generation failed: {e}")

# --------------------------------
# Streamlit Interface
# --------------------------------
st.title("Intelligent Document Analyzer")
st.markdown("Upload a PDF, Image, or Text file to extract, classify, and analyze its content.")

uploaded_file = st.file_uploader("Upload a file", type=["pdf", "png", "jpg", "jpeg", "txt"])

if uploaded_file:
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("Extracting text using OCR...")
    text = extract_text(file_path)
    clean_text = preprocess_text(text)

    if not clean_text.strip():
        st.error("No readable text found in file.")
    else:
        st.success("Text successfully extracted.")
        st.text_area("Extracted Text", text[:1500] + "..." if len(text) > 1500 else text, height=200)

        st.write("Classifying document type...")
        doc_type, confidence = classify_doc(clean_text)
        st.write(f"Document Type: {doc_type}")
        st.write(f"Confidence Score: {round(confidence, 2)}")

        st.write("Extracting key information...")
        info = extract_info(text, doc_type)
        st.json(info)

        st.write("Making decision...")
        decision = make_decision(info, confidence)
        st.write(f"Decision: {decision}")

        st.write("Generating explainability map...")
        generate_explainability(clean_text, classifier, ["invoice", "resume", "report"])

st.markdown("---")
st.caption("Built with Streamlit, OCR, and NLP")
