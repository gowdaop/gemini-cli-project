from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Set
import logging

from ..schemas.analysis import UploadResponse, OCRText, OCRBlock, PageSpan, ErrorResponse
from ..services import ocr
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/upload",
    tags=["upload"],
    responses={
        415: {"model": ErrorResponse, "description": "Unsupported Media Type"},
        413: {"model": ErrorResponse, "description": "Request Entity Too Large"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    }
)

# Allowed file types based on config
ALLOWED_MIME_TYPES: Set[str] = set(settings.ALLOWED_MIME_TYPES)

def validate_file(file: UploadFile) -> None:
    """Validate uploaded file type and size"""
    # Check MIME type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Allowed types: {', '.join(ALLOWED_MIME_TYPES)}"
        )
    
    # Check file size if available
    if hasattr(file, 'size') and file.size and file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({file.size} bytes) exceeds maximum allowed size ({settings.MAX_FILE_SIZE} bytes)"
        )

@router.post("/", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and extract text from legal document
    
    Supports PDF, DOCX, PNG, and JPEG files up to 20MB.
    Returns structured OCR data with text blocks and their locations.
    """
    try:
        # Validate file
        validate_file(file)
        
        logger.info(f"Processing upload: {file.filename} ({file.content_type})")
        
        # Extract text using OCR service
        ocr_result = await ocr.extract_text(file)
        
        # Convert to Pydantic model for validation
        ocr_text = OCRText(
            full_text=ocr_result.get("full_text", ""),
            blocks=[
                OCRBlock(
                    text=block.get("text", ""),
                    span=PageSpan(
                        page=block.get("span", {}).get("page", 1),
                        start_line=block.get("span", {}).get("start_line", 1),
                        end_line=block.get("span", {}).get("end_line", 1)
                    )
                )
                for block in ocr_result.get("blocks", [])
            ]
        )
        
        logger.info(f"Successfully processed {file.filename}: {len(ocr_text.blocks)} blocks extracted")
        
        return UploadResponse(ocr=ocr_text)
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Upload processing failed for {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {str(e)}"
        )

@router.get("/health")
async def upload_health():
    """Health check for upload service"""
    return {"status": "healthy", "service": "upload"}
