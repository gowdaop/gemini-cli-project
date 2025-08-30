from fastapi import FastAPI, Depends, HTTPException, Request, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import time
import logging
import uvicorn
from pydantic import BaseModel
# Fix import paths
from .schemas.analysis import extract_text, analyze_document, answer_with_vertex
from .config import settings

from .config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add missing Pydantic models
class ChatRequest(BaseModel):
    question: str
    context: Optional[str] = ""

class AnalyzeRequest(BaseModel):
    ocr: Dict[str, Any]
    top_k: int = 5

class UploadResponse(BaseModel):
    ocr: Dict[str, Any]

class AnalyzeResponse(BaseModel):
    clauses: List[Dict[str, Any]]
    risks: List[Dict[str, Any]]
    summary: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []
    confidence: Optional[float] = None
class ErrorResponse(BaseModel):
    detail: str
    code: Optional[str] = None
# API Key authentication
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

async def require_api_key(x_api_key: Optional[str] = Depends(api_key_header)):
    """Global API key validation dependency"""
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key header 'x-api-key' is required"
        )
    
    if x_api_key != settings.API_KEY:
        logger.warning(f"Invalid API key attempted: {x_api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return True

# Lifespan events for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events"""
    # Startup
    logger.info("Starting Legal Document Simplifier API...")
    logger.info(f"Environment: {'Development' if settings.DEBUG else 'Production'}")
    logger.info(f"GCP Project: {settings.GCP_PROJECT_ID}")
    logger.info(f"Milvus Host: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
    
    # Validate required configurations
    if not settings.GCP_PROJECT_ID:
        logger.warning("GCP_PROJECT_ID not set - Document AI may not work")
    if not settings.DOCAI_PROCESSOR_ID:
        logger.warning("DOCAI_PROCESSOR_ID not set - OCR will use fallback")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Legal Document Simplifier API...")

# Create FastAPI application
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"] if settings.DEBUG else ["*.run.app", "localhost"]
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
    max_age=3600,
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests and response times"""
    start_time = time.time()
    
    # Log request
    logger.info(f"{request.method} {request.url.path} - Client: {request.client.host}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"{request.method} {request.url.path} - "
            f"Error: {str(e)} - "
            f"Time: {process_time:.3f}s"
        )
        raise

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            detail="Internal server error. Please try again later."
        ).dict()
    )

# Health check endpoint (no auth required)
@app.get("/healthz", tags=["health"])
async def health_check():
    """Health check endpoint for Cloud Run"""
    return {
        "status": "ok",
        "version": settings.API_VERSION,
        "timestamp": time.time()
    }

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """API root endpoint"""
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "description": settings.API_DESCRIPTION,
        "docs_url": "/docs" if settings.DEBUG else None
    }

# Fixed endpoints with correct function names and imports
@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    current_user: str = Depends(require_api_key)  # Fixed: was verify_api_key
):
    """Upload and process document"""
    try:
        # Use your analysis functions
        ocr_result = await extract_text(file)
        return {"ocr": ocr_result}  # Matches test expectation
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_document_endpoint(
    request: AnalyzeRequest,
    current_user: str = Depends(require_api_key)  # Fixed: was verify_api_key
):
    """Analyze document for clauses and risks"""
    try:
        analysis_result = analyze_document(request.ocr, request.top_k)
        return analysis_result  # Matches test expectation
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    current_user: str = Depends(require_api_key)  # Fixed: was verify_api_key
):
    """Chat with legal document using RAG"""
    try:
        chat_result = answer_with_vertex(request.question)
        return chat_result  # Matches test expectation
        
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

# Custom OpenAPI schema
def custom_openapi():
    """Generate custom OpenAPI schema"""
    if app.openapi_schema:
        return app.openapi_schema
        
    openapi_schema = get_openapi(
        title=settings.API_TITLE,
        version=settings.API_VERSION,
        description=settings.API_DESCRIPTION,
        routes=app.routes,
    )
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "x-api-key"
        }
    }
    
    # Add security requirement to all paths
    for path in openapi_schema["paths"].values():
        for method in path.values():
            if isinstance(method, dict) and "tags" in method:
                if "health" not in method["tags"] and "root" not in method["tags"]:
                    method["security"] = [{"ApiKeyAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.MAX_WORKERS,
        timeout_keep_alive=settings.KEEP_ALIVE_TIMEOUT,
        timeout_worker=settings.WORKER_TIMEOUT,
        log_level="info"
    )
