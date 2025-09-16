from pydantic_settings import BaseSettings
from pydantic import field_validator, ValidationError
from typing import List, Optional, Union
import os
import json
import logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    # API Configuration
    API_TITLE: str = "Legal Document Simplifier API"
    API_VERSION: str = "v1"
    API_DESCRIPTION: str = "AI-powered legal document analysis with risk scoring and RAG-based Q&A"
    DEBUG: bool = False

    # Authentication
    API_KEY: str = "dev-key"

    # CORS Configuration - Using Union to prevent automatic JSON parsing
    CORS_ORIGINS: Union[str, List[str]] = "http://localhost:3000,https://localhost:3000"
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: Union[str, List[str]] = "GET,POST,PUT,DELETE,OPTIONS"
    CORS_ALLOW_HEADERS: Union[str, List[str]] = "*"

    # GCP Configuration
    GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "")
    GCP_LOCATION: str = "us"
    DOCAI_PROCESSOR_ID: str = "bd8d8a755148c6a1"
    DOCAI_PROCESSOR_VERSION: str = "rc"
    GOOGLE_APPLICATION_CREDENTIALS: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    # Vertex AI Configuration
    VERTEX_MODEL: str = os.getenv("VERTEX_MODEL", "gemini-2.0-flash")
    VERTEX_LOCATION: str = os.getenv("VERTEX_LOCATION", "us")
    VERTEX_MAX_TOKENS: int = int(os.getenv("VERTEX_MAX_TOKENS", "2048"))
    VERTEX_TEMPERATURE: float = float(os.getenv("VERTEX_TEMPERATURE", "0.3"))
    VERTEX_SEARCH_ENGINE_ID: str = os.getenv("VERTEX_SEARCH_ENGINE_ID", "legal-document-search_1757848376569")
    VERTEX_SEARCH_PROJECT_ID: str = os.getenv("VERTEX_SEARCH_PROJECT_ID", "legal-470717")
    # Milvus Configuration
    MILVUS_HOST: str = "milvus-standalone"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "legal_chunks_v1"
    MILVUS_TIMEOUT: int = 30

    # HuggingFace Configuration
    HF_MODEL_NAME: str = "law-ai/InLegalBERT"
    HF_CACHE_DIR: str = "/tmp/hf_cache"

    # File Upload Limits
    MAX_FILE_SIZE: int = 20 * 1024 * 1024  # 20MB
    MAX_PAGES: int = 15
    ALLOWED_MIME_TYPES: Union[str, List[str]] = "application/pdf,image/png,image/jpeg,application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    # Security
    SECRET_MANAGER_PROJECT: Optional[str] = None
    USE_SECRET_MANAGER: bool = False

    # Performance
    WORKER_TIMEOUT: int = 300
    KEEP_ALIVE_TIMEOUT: int = 5
    MAX_WORKERS: int = 4

    @field_validator('CORS_ORIGINS', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins - handles string, list, or JSON"""
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return ["http://localhost:3000"]
            
            # Try JSON first (for backwards compatibility)
            if v.startswith('[') and v.endswith(']'):
                try:
                    parsed = json.loads(v)
                    return parsed if isinstance(parsed, list) else [str(parsed)]
                except:
                    pass
            
            # Handle comma-separated values
            return [origin.strip() for origin in v.split(',') if origin.strip()]
        
        return ["http://localhost:3000"]

    @field_validator('CORS_ALLOW_METHODS', mode='before')
    @classmethod
    def parse_cors_methods(cls, v):
        """Parse CORS methods"""
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
            
            if v.startswith('[') and v.endswith(']'):
                try:
                    parsed = json.loads(v)
                    return parsed if isinstance(parsed, list) else [str(parsed)]
                except:
                    pass
            
            return [method.strip() for method in v.split(',') if method.strip()]
        
        return ["GET", "POST", "PUT", "DELETE", "OPTIONS"]

    @field_validator('CORS_ALLOW_HEADERS', mode='before')
    @classmethod
    def parse_cors_headers(cls, v):
        """Parse CORS headers"""
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return ["*"]
            
            if v.startswith('[') and v.endswith(']'):
                try:
                    parsed = json.loads(v)
                    return parsed if isinstance(parsed, list) else [str(parsed)]
                except:
                    pass
            
            return [header.strip() for header in v.split(',') if header.strip()]
        
        return ["*"]

    @field_validator('ALLOWED_MIME_TYPES', mode='before')
    @classmethod
    def parse_allowed_mime_types(cls, v):
        """Parse allowed MIME types"""
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return ["application/pdf", "image/png", "image/jpeg", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
            
            if v.startswith('[') and v.endswith(']'):
                try:
                    parsed = json.loads(v)
                    return parsed if isinstance(parsed, list) else [str(parsed)]
                except:
                    pass
            
            return [mime.strip() for mime in v.split(',') if mime.strip()]
        
        return ["application/pdf", "image/png", "image/jpeg", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]

    @field_validator('GCP_PROJECT_ID', mode='before')
    @classmethod
    def validate_gcp_project(cls, v):
        """Get GCP project from environment if not provided"""
        if not v and os.getenv('GOOGLE_CLOUD_PROJECT'):
            return os.getenv('GOOGLE_CLOUD_PROJECT')
        return v or ""

    @field_validator('DEBUG', mode='before')
    @classmethod
    def parse_debug(cls, v):
        """Parse DEBUG from string boolean"""
        if isinstance(v, str):
            return v.lower() in ('true', '1', 'yes', 'on')
        return bool(v)

    @field_validator('USE_SECRET_MANAGER', mode='before')
    @classmethod
    def parse_use_secret_manager(cls, v):
        """Parse USE_SECRET_MANAGER from string boolean"""
        if isinstance(v, str):
            return v.lower() in ('true', '1', 'yes', 'on')
        return bool(v)

    def load_from_secret_manager(self):
        """Load sensitive settings from Google Secret Manager"""
        if not self.USE_SECRET_MANAGER or not self.SECRET_MANAGER_PROJECT:
            return

        try:
            from google.cloud import secretmanager
            client = secretmanager.SecretManagerServiceClient()
            
            # Load API key from Secret Manager
            try:
                secret_name = f"projects/{self.SECRET_MANAGER_PROJECT}/secrets/api-key/versions/latest"
                response = client.access_secret_version(request={"name": secret_name})
                self.API_KEY = response.payload.data.decode("UTF-8")
                logger.info("API key loaded from Secret Manager")
            except Exception as e:
                logger.warning(f"Could not load API key from Secret Manager: {e}")

            # Load Document AI processor ID
            try:
                docai_secret = f"projects/{self.SECRET_MANAGER_PROJECT}/secrets/docai-processor-id/versions/latest"
                response = client.access_secret_version(request={"name": docai_secret})
                self.DOCAI_PROCESSOR_ID = response.payload.data.decode("UTF-8")
                logger.info("Document AI processor ID loaded from Secret Manager")
            except Exception as e:
                logger.warning(f"Could not load Document AI processor ID from Secret Manager: {e}")

        except ImportError:
            logger.warning("Google Cloud Secret Manager not available")
        except Exception as e:
            logger.error(f"Failed to initialize Secret Manager client: {e}")

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True

# Initialize settings with comprehensive error handling
def create_settings():
    """Create settings instance with fallback"""
    try:
        settings_instance = Settings()
        
        # Convert Union types to Lists after initialization
        if isinstance(settings_instance.CORS_ORIGINS, str):
            settings_instance.CORS_ORIGINS = settings_instance.parse_cors_origins(settings_instance.CORS_ORIGINS)
        if isinstance(settings_instance.CORS_ALLOW_METHODS, str):
            settings_instance.CORS_ALLOW_METHODS = settings_instance.parse_cors_methods(settings_instance.CORS_ALLOW_METHODS)
        if isinstance(settings_instance.CORS_ALLOW_HEADERS, str):
            settings_instance.CORS_ALLOW_HEADERS = settings_instance.parse_cors_headers(settings_instance.CORS_ALLOW_HEADERS)
        if isinstance(settings_instance.ALLOWED_MIME_TYPES, str):
            settings_instance.ALLOWED_MIME_TYPES = settings_instance.parse_allowed_mime_types(settings_instance.ALLOWED_MIME_TYPES)
        
        # Load from Secret Manager if configured
        if settings_instance.USE_SECRET_MANAGER:
            settings_instance.load_from_secret_manager()
        
        logger.info("Settings loaded successfully")
        logger.debug(f"CORS Origins: {settings_instance.CORS_ORIGINS}")
        logger.debug(f"Allowed MIME types: {settings_instance.ALLOWED_MIME_TYPES}")
        
        return settings_instance
        
    except ValidationError as e:
        logger.error(f"Settings validation failed: {e}")
        # Create emergency fallback settings
        fallback = Settings(
            API_KEY="emergency-fallback-key",
            GCP_PROJECT_ID="",
            DOCAI_PROCESSOR_ID="",
            DEBUG=True,
            CORS_ORIGINS=["http://localhost:3000"],
            CORS_ALLOW_METHODS=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            CORS_ALLOW_HEADERS=["*"],
            ALLOWED_MIME_TYPES=["application/pdf", "image/png", "image/jpeg", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
        )
        logger.warning("Using emergency fallback settings")
        return fallback
        
    except Exception as e:
        logger.error(f"Settings initialization failed completely: {e}")
        raise

# Global settings instance
settings = create_settings()
