from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List, Optional
import os
from google.cloud import secretmanager
import json


class Settings(BaseSettings):
    # API Configuration
    API_TITLE: str = "Legal Document Simplifier API"
    API_VERSION: str = "v1"
    API_DESCRIPTION: str = "AI-powered legal document analysis with risk scoring and RAG-based Q&A"
    DEBUG: bool = False
    
    # Authentication
    API_KEY: str = "dev-key"
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "https://localhost:3000"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # GCP Configuration
    GCP_PROJECT_ID: str = ""
    GCP_LOCATION: str = "us-central1"
    DOCAI_PROCESSOR_ID: str = ""
    DOCAI_PROCESSOR_VERSION: str = "rc"
    
    # Vertex AI Configuration
    VERTEX_MODEL: str = "gemini-1.5-pro"
    VERTEX_LOCATION: str = "us-central1"
    VERTEX_MAX_TOKENS: int = 8192
    VERTEX_TEMPERATURE: float = 0.1
    
    # Milvus Configuration
    MILVUS_HOST: str = "milvus-standalone"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "legal_chunks_v1"
    MILVUS_TIMEOUT: int = 30
    
    # HuggingFace Configuration
    HF_MODEL_NAME: str = "nlpaueb/legal-bert-base-uncased"
    HF_CACHE_DIR: str = "/tmp/hf_cache"
    
    # File Upload Limits
    MAX_FILE_SIZE: int = 20 * 1024 * 1024  # 20MB
    MAX_PAGES: int = 15
    ALLOWED_MIME_TYPES: List[str] = [
        "application/pdf",
        "image/png", 
        "image/jpeg",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]
    
    # Security
    SECRET_MANAGER_PROJECT: Optional[str] = None
    USE_SECRET_MANAGER: bool = False
    
    # Performance
    WORKER_TIMEOUT: int = 300
    KEEP_ALIVE_TIMEOUT: int = 5
    MAX_WORKERS: int = 4
    
    @field_validator('CORS_ORIGINS', mode='before')
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @field_validator('GCP_PROJECT_ID')
    def validate_gcp_project(cls, v):
        if not v and os.getenv('GOOGLE_CLOUD_PROJECT'):
            return os.getenv('GOOGLE_CLOUD_PROJECT')
        return v
    
    def load_from_secret_manager(self):
        """Load sensitive settings from Google Secret Manager"""
        if not self.USE_SECRET_MANAGER or not self.SECRET_MANAGER_PROJECT:
            return
            
        try:
            client = secretmanager.SecretManagerServiceClient()
            
            # Load API key from Secret Manager
            secret_name = f"projects/{self.SECRET_MANAGER_PROJECT}/secrets/api-key/versions/latest"
            response = client.access_secret_version(request={"name": secret_name})
            self.API_KEY = response.payload.data.decode("UTF-8")
            
            # Load other secrets as needed
            docai_secret = f"projects/{self.SECRET_MANAGER_PROJECT}/secrets/docai-processor-id/versions/latest"
            response = client.access_secret_version(request={"name": docai_secret})
            self.DOCAI_PROCESSOR_ID = response.payload.data.decode("UTF-8")
            
        except Exception as e:
            print(f"Warning: Could not load secrets from Secret Manager: {e}")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()

# Load from Secret Manager in production
if settings.USE_SECRET_MANAGER:
    settings.load_from_secret_manager()
