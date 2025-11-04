"""
FastAPI REST API for Document AI System
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from pathlib import Path
import shutil
import uuid
from loguru import logger
import asyncio

from ..pipeline import DocumentAIPipeline
from ..config import load_config

# Initialize app
app = FastAPI(
    title="Document AI System",
    description="End-to-End AI System for Intelligent Document Understanding",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[DocumentAIPipeline] = None
config = None

# Models
class ProcessingResponse(BaseModel):
    job_id: str
    status: str
    message: str


class DocumentResult(BaseModel):
    document_type: str
    fields_extracted: Dict
    decision: str
    confidence_score: float
    validation: Dict
    reasoning: List[str]
    explainability_map: str
    processing_time_seconds: float
    metadata: Dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class JobRequirements(BaseModel):
    required_skills: List[str] = []
    min_experience: int = 0
    preferred_education: List[str] = []


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    global pipeline, config
    
    logger.info("Starting Document AI API...")
    
    try:
        config = load_config()
        
        # Create necessary directories
        Path(config.api.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(config.api.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize pipeline
        pipeline = DocumentAIPipeline()
        
        logger.info("API started successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return {
        "status": "healthy",
        "model_loaded": pipeline.model is not None,
        "device": str(pipeline.device)
    }


# Document upload and processing endpoint
@app.post("/process")
async def process_document(
    file: UploadFile = File(...),
    generate_explanations: bool = True,
    background_tasks: BackgroundTasks = None
):
    """
    Upload and process a document and return the full result payload
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in config.api.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not supported. Allowed: {config.api.allowed_extensions}"
        )
    
    # Validate file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > config.api.max_file_size:
        raise HTTPException(
            status_code=400,
            detail=f"File size {file_size} exceeds maximum {config.api.max_file_size}"
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    upload_path = Path(config.api.upload_dir) / f"{job_id}_{file.filename}"
    
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File uploaded: {upload_path}")
        
        # Process document synchronously (for demo)
        result = pipeline.process_document(
            str(upload_path),
            generate_explanations=generate_explanations
        )
        
        # Attach job id for traceability
        result["job_id"] = job_id
        
        # Save result
        result_path = Path(config.api.output_dir) / f"{job_id}_result.json"
        pipeline.save_result(result, str(result_path))
        
        return result
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Get processing result
@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """Get processing result by job ID"""
    result_path = Path(config.api.output_dir) / f"{job_id}_result.json"
    
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    
    import json
    with open(result_path, 'r') as f:
        result = json.load(f)
    
    return result


# Get explainability visualization
@app.get("/visualization/{job_id}/{viz_type}")
async def get_visualization(job_id: str, viz_type: str):
    """
    Get explainability visualization
    
    Args:
        job_id: Job ID
        viz_type: Type of visualization (summary, attention, importance)
    """
    viz_map = {
        'summary': f"{job_id}_summary.png",
        'attention': f"{job_id}_attention.png",
        'importance': f"{job_id}_importance.png"
    }
    
    if viz_type not in viz_map:
        raise HTTPException(status_code=400, detail="Invalid visualization type")
    
    viz_path = Path(config.api.output_dir) / viz_map[viz_type]
    
    if not viz_path.exists():
        raise HTTPException(status_code=404, detail="Visualization not found")
    
    return FileResponse(viz_path, media_type="image/png")


# Batch processing endpoint
@app.post("/batch-process")
async def batch_process(files: List[UploadFile] = File(...)):
    """Process multiple documents in batch"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    job_ids = []
    file_paths = []
    
    # Save all files
    for file in files:
        job_id = str(uuid.uuid4())
        upload_path = Path(config.api.upload_dir) / f"{job_id}_{file.filename}"
        
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        job_ids.append(job_id)
        file_paths.append(str(upload_path))
    
    # Process batch
    try:
        results = pipeline.batch_process(file_paths, generate_explanations=False)
        
        # Save results
        for job_id, result in zip(job_ids, results):
            result_path = Path(config.api.output_dir) / f"{job_id}_result.json"
            pipeline.save_result(result, str(result_path))
        
        return {
            "job_ids": job_ids,
            "status": "completed",
            "total_documents": len(files)
        }
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Resume ranking endpoint
@app.post("/rank-resumes")
async def rank_resumes(
    files: List[UploadFile] = File(...),
    job_requirements: Optional[JobRequirements] = None
):
    """
    Process and rank multiple resumes
    
    Args:
        files: List of resume files
        job_requirements: Job requirements for ranking
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    file_paths = []
    
    # Save all resume files
    for file in files:
        job_id = str(uuid.uuid4())
        upload_path = Path(config.api.upload_dir) / f"{job_id}_{file.filename}"
        
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_paths.append(str(upload_path))
    
    # Rank resumes
    try:
        requirements = job_requirements.dict() if job_requirements else None
        ranked_resumes = pipeline.rank_resumes(file_paths, requirements)
        
        return {
            "total_resumes": len(ranked_resumes),
            "ranked_resumes": ranked_resumes
        }
        
    except Exception as e:
        logger.error(f"Resume ranking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Invoice validation endpoint
@app.post("/validate-invoice")
async def validate_invoice(file: UploadFile = File(...)):
    """Validate invoice document"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Save file
    job_id = str(uuid.uuid4())
    upload_path = Path(config.api.upload_dir) / f"{job_id}_{file.filename}"
    
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process
    try:
        result = pipeline.process_document(str(upload_path), generate_explanations=True)
        
        if result['document_type'] != 'invoice':
            return {
                "warning": "Document does not appear to be an invoice",
                "detected_type": result['document_type']
            }
        
        return {
            "validation": result['validation'],
            "decision": result['decision'],
            "confidence": result['confidence_score'],
            "fields": result['fields_extracted'],
            "rule_violations": result['rule_violations']
        }
        
    except Exception as e:
        logger.error(f"Invoice validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model info endpoint
@app.get("/model-info")
async def get_model_info():
    """Get information about loaded model"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return {
        "model_name": config.model.name,
        "document_types": config.document_types,
        "device": str(pipeline.device),
        "ocr_engines": config.ocr.engines,
        "max_file_size": config.api.max_file_size,
        "allowed_extensions": config.api.allowed_extensions
    }


# Root endpoint
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "Document AI System API",
        "version": "1.0.0",
        "description": "End-to-End AI System for Intelligent Document Understanding",
        "endpoints": {
            "/health": "Health check",
            "/process": "Process single document",
            "/result/{job_id}": "Get processing result",
            "/batch-process": "Process multiple documents",
            "/rank-resumes": "Rank multiple resumes",
            "/validate-invoice": "Validate invoice",
            "/model-info": "Get model information"
        },
        "streamlit_ui": "http://localhost:8501"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
