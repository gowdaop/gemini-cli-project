from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager
from typing import Optional
import time
import logging
import uvicorn
from vertexai import init as vertex_init

from .config import settings
from .routers import upload, analyze, chat
from .schemas.analysis import ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# API Key authentication
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

async def require_api_key(x_api_key: Optional[str] = Depends(api_key_header)):
    """Global API key validation dependency"""
    
    # TEMPORARY DEBUG LOGGING - REMOVE AFTER FIXING
    print(f"🔍 Expected API key: '{settings.API_KEY}'")
    print(f"🔍 Received API key: '{x_api_key}'")
    print(f"🔍 Keys match: {x_api_key == settings.API_KEY}")
    
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

# ✅ FIXED: Include each router only ONCE
app.include_router(
    upload.router, 
    dependencies=[Depends(require_api_key)]
)
app.include_router(
    analyze.router,
    dependencies=[Depends(require_api_key)]
)
app.include_router(
    chat.router,
    dependencies=[Depends(require_api_key)]
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
@app.get("/healthz", tags=["health"])
async def health_check():
    """Health check endpoint for Cloud Run"""
    return {
        "status": "ok",
        "version": settings.API_VERSION,
        "timestamp": time.time()
    }

# 🔥 ADD THESE NEW ENDPOINTS HERE 🔥

@app.get("/health/detailed", tags=["health"])
async def detailed_health_check():
    """Detailed health check with full service diagnostics"""
    try:
        from .services import rag
        
        service = await rag.get_rag_service()
        stats = await service.get_service_stats()
        
        return {
            "api": {
                "status": "healthy",
                "version": settings.API_VERSION,
                "debug_mode": settings.DEBUG
            },
            "environment": {
                "gcp_project": settings.GCP_PROJECT_ID or "not_configured",
                "milvus_host": settings.MILVUS_HOST,
                "milvus_collection": settings.MILVUS_COLLECTION
            },
            "services": stats,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }

@app.get("/health/config", tags=["health"])
async def config_health_check():
    """Check configuration health (sensitive info masked)"""
    return {
        "google_api_key": "configured" if getattr(settings, 'GOOGLE_API_KEY', None) else "missing",
        "google_search_engine_id": "configured" if getattr(settings, 'GOOGLE_SEARCH_ENGINE_ID', None) else "missing",
        "gcp_project": "configured" if settings.GCP_PROJECT_ID else "missing",
        "milvus_collection": settings.MILVUS_COLLECTION,
        "vertex_model": getattr(settings, 'VERTEX_MODEL', 'gemini-2.0-flash'),
        "debug_mode": settings.DEBUG,
        "docai_processor": "configured" if settings.DOCAI_PROCESSOR_ID else "missing"
    }

@app.get("/health/services", tags=["health"])
async def all_services_health():
    """Check health of all integrated services"""
    try:
        from .services import rag
        
        service = await rag.get_rag_service()
        
        # Check Milvus
        milvus_status = "healthy" if service.milvus_client.connected else "unhealthy"
        
        # Check Vertex AI
        vertex_status = "healthy" if service.vertex_client.initialized else "unhealthy"
        
        # Check Web Search
        web_status = "healthy" if service.web_search_service.initialized else "unhealthy"
        
        # Overall status
        overall_status = "healthy" if all([
            service.milvus_client.connected,
            service.vertex_client.initialized,
            service.web_search_service.initialized
        ]) else "degraded"
        
        return {
            "overall_status": overall_status,
            "services": {
                "milvus": {
                    "status": milvus_status,
                    "connected": service.milvus_client.connected,
                    "collection": settings.MILVUS_COLLECTION
                },
                "vertex_ai": {
                    "status": vertex_status,
                    "initialized": service.vertex_client.initialized,
                    "model": getattr(settings, 'VERTEX_MODEL', 'gemini-2.0-flash')
                },
                "web_search": {
                    "status": web_status,
                    "initialized": service.web_search_service.initialized,
                    "project_configured": bool(service.web_search_service.project_id),    # ✅ Vertex AI Search
                    "engine_configured": bool(service.web_search_service.engine_id)
                }
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "overall_status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }

@app.post("/health/test-search", tags=["health"])
async def test_web_search(query: str = "contract liability"):
    """Test web search with a specific query"""
    try:
        from .services import rag
        
        service = await rag.get_rag_service()
        
        # Test web search
        web_results = await service.web_search_service.search_web(query, num_results=3)
        
        # Test RAG context conversion
        rag_contexts = service.web_search_service.convert_to_rag_context(web_results)
        
        return {
            "status": "success",
            "query": query,
            "web_results_count": len(web_results),
            "rag_contexts_count": len(rag_contexts),
            "sample_results": [
                {
                    "title": result.get("title", "")[:100],
                    "source": result.get("source", ""),
                    "link": result.get("link", "")[:100]
                } for result in web_results[:2]
            ] if web_results else []
        }
        
    except Exception as e:
        return {
            "status": "error",
            "query": query,
            "error": str(e)
        }
# Health check endpoint (no auth required)
# Health check endpoint (no auth required)
@app.get("/healthz", tags=["health"])
async def health_check():
    """Enhanced health check endpoint for Cloud Run"""
    try:
        # Import here to avoid circular imports
        from .services import rag
        
        # Get service stats
        service_stats = await rag.health_check()
        
        return {
            "status": "ok",
            "version": settings.API_VERSION,
            "timestamp": time.time(),
            "services": {
                "rag_service": service_stats.get("status", "unknown"),
                "milvus": service_stats.get("milvus_connected", False),
                "vertex_ai": service_stats.get("vertex_initialized", False),
                "web_search": service_stats.get("web_search_initialized", False)
            }
        }
    except Exception as e:
        return {
            "status": "degraded",
            "version": settings.API_VERSION,
            "timestamp": time.time(),
            "error": str(e)
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
    
    # Add security requirement to all paths except health and root
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
