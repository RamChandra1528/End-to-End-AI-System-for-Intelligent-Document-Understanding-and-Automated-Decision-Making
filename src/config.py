"""
Configuration loader and utilities for the Document AI System
"""
import yaml
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    name: str = "microsoft/layoutlmv3-base"
    num_labels: int = 3
    max_length: int = 512
    image_size: int = 224


class OCRConfig(BaseModel):
    engines: list[str] = ["tesseract", "easyocr"]
    languages: list[str] = ["en"]
    confidence_threshold: float = 0.6


class TrainingConfig(BaseModel):
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 2
    mixed_precision: bool = True


class ReasoningConfig(BaseModel):
    invoice: Dict[str, Any] = Field(default_factory=dict)
    resume: Dict[str, Any] = Field(default_factory=dict)
    report: Dict[str, Any] = Field(default_factory=dict)


class ExplainabilityConfig(BaseModel):
    use_attention: bool = True
    use_shap: bool = True
    heatmap_resolution: list[int] = [800, 600]
    top_k_features: int = 10


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    max_file_size: int = 10485760
    allowed_extensions: list[str] = [".pdf", ".png", ".jpg", ".jpeg", ".tiff"]
    upload_dir: str = "uploads"
    output_dir: str = "outputs"


class RAGConfig(BaseModel):
    enabled: bool = True
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store: str = "chromadb"
    top_k_results: int = 3


class Config(BaseSettings):
    model_config = {"extra": "ignore"}  # Allow extra fields from YAML
    
    model: ModelConfig = Field(default_factory=ModelConfig)
    document_types: list[str] = ["invoice", "resume", "report"]
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert nested dicts to Pydantic models
        if 'model' in config_dict:
            config_dict['model'] = ModelConfig(**config_dict['model'])
        if 'ocr' in config_dict:
            config_dict['ocr'] = OCRConfig(**config_dict['ocr'])
        if 'training' in config_dict:
            config_dict['training'] = TrainingConfig(**config_dict['training'])
        if 'reasoning' in config_dict:
            config_dict['reasoning'] = ReasoningConfig(**config_dict['reasoning'])
        if 'explainability' in config_dict:
            config_dict['explainability'] = ExplainabilityConfig(**config_dict['explainability'])
        if 'api' in config_dict:
            config_dict['api'] = APIConfig(**config_dict['api'])
        if 'rag' in config_dict:
            config_dict['rag'] = RAGConfig(**config_dict['rag'])
            
        return cls(**config_dict)


def load_config(config_path: str = None) -> Config:
    """Load configuration from file or use defaults"""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    
    if Path(config_path).exists():
        return Config.from_yaml(config_path)
    else:
        return Config()
