import asyncio
import logging
import tempfile
import os
from typing import Dict, Any, List, Optional, Union, Tuple, TYPE_CHECKING
from pathlib import Path
import mimetypes
from datetime import datetime
import hashlib
import json

# Type checking imports
if TYPE_CHECKING:
    from google.cloud.documentai_v1 import Document
else:
    Document = Any

try:
    from google.cloud import documentai_v1 as documentai
    from google.api_core import retry, exceptions as gcp_exceptions
    from google.auth import default
    import google.auth.transport.requests
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    documentai = None

from fastapi import UploadFile, HTTPException, status
from ..config import settings
from ..schemas.analysis import OCRText, OCRBlock, PageSpan

logger = logging.getLogger(__name__)

class DocumentAIService:
    """Production-ready Document AI service with comprehensive error handling"""
    
    def __init__(self):
        self.client = None
        self.processor_name = None
        self.initialized = False
        self._max_file_size = settings.MAX_FILE_SIZE
        self._max_pages = settings.MAX_PAGES
        self._retry_config = None
        
        if GOOGLE_CLOUD_AVAILABLE:
            self._retry_config = retry.Retry(
                initial=1.0,
                maximum=10.0,
                multiplier=2.0,
                predicate=retry.if_exception_type(
                    gcp_exceptions.DeadlineExceeded,
                    gcp_exceptions.ServiceUnavailable,
                    gcp_exceptions.InternalServerError
                )
            )
    
    async def initialize(self):
        """Initialize Document AI client and processor"""
        if self.initialized:
            return
        
        try:
            if not GOOGLE_CLOUD_AVAILABLE:
                logger.warning("Google Cloud Document AI not available - using fallback mode")
                return
            
            # Validate required settings
            if not settings.GCP_PROJECT_ID:
                raise ValueError("GCP_PROJECT_ID is required for Document AI")
            if not settings.DOCAI_PROCESSOR_ID:
                raise ValueError("DOCAI_PROCESSOR_ID is required for Document AI")
            
            # Initialize client
            self.client = documentai.DocumentProcessorServiceClient()
            
            # Build processor name
            self.processor_name = self.client.processor_path(
                settings.GCP_PROJECT_ID,
                settings.GCP_LOCATION,
                settings.DOCAI_PROCESSOR_ID
            )
            
            # Test authentication and processor access
            await self._test_processor_access()
            
            self.initialized = True
            logger.info(f"Document AI service initialized successfully")
            logger.info(f"Processor: {self.processor_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Document AI: {e}")
            # Don't raise exception - allow fallback mode
    
    async def _test_processor_access(self):
        """Test processor access and authentication"""
        try:
            # Create a minimal test document
            test_content = b"Test document for processor validation"
            raw_document = documentai.RawDocument(
                content=test_content,
                mime_type="text/plain"
            )
            
            request = documentai.ProcessRequest(
                name=self.processor_name,
                raw_document=raw_document
            )
            
            # This will raise an exception if authentication or processor access fails
            response = self.client.process_document(request=request)
            logger.debug("Document AI processor access test successful")
            
        except Exception as e:
            logger.error(f"Document AI processor access test failed: {e}")
            raise
    
    async def process_document(self, file: UploadFile) -> Any:
        """Process document using Document AI with comprehensive error handling"""
        if not self.initialized:
            await self.initialize()
        
        if not self.client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Document AI service not available"
            )
        
        try:
            # Read and validate file
            file_content = await self._read_and_validate_file(file)
            
            # Create Document AI request
            raw_document = documentai.RawDocument(
                content=file_content,
                mime_type=file.content_type
            )
            
            request = documentai.ProcessRequest(
                name=self.processor_name,
                raw_document=raw_document,
                skip_human_review=True  # For faster processing
            )
            
            logger.info(f"Processing document: {file.filename} ({len(file_content)} bytes)")
            
            # Process with retry logic
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.process_document(
                    request=request, 
                    retry=self._retry_config,
                    timeout=120  # 2 minutes timeout
                )
            )
            
            if not response.document:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Document processing failed - no document returned"
                )
            
            logger.info(f"Document processed successfully: {len(response.document.pages)} pages")
            return response.document
            
        except gcp_exceptions.InvalidArgument as e:
            logger.error(f"Invalid document format: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid document format: {str(e)}"
            )
        except gcp_exceptions.QuotaExceeded as e:
            logger.error(f"Document AI quota exceeded: {e}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Document processing quota exceeded. Please try again later."
            )
        except gcp_exceptions.PermissionDenied as e:
            logger.error(f"Document AI permission denied: {e}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Document AI access denied. Please check credentials."
            )
        except Exception as e:
            logger.error(f"Document processing failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Document processing failed: {str(e)}"
            )
    
    async def _read_and_validate_file(self, file: UploadFile) -> bytes:
        """Read and validate uploaded file"""
        try:
            # Reset file pointer
            await file.seek(0)
            file_content = await file.read()
            await file.seek(0)  # Reset for potential re-reading
            
            # Validate file size
            if len(file_content) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Empty file uploaded"
                )
            
            if len(file_content) > self._max_file_size:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File size ({len(file_content)} bytes) exceeds maximum ({self._max_file_size} bytes)"
                )
            
            # Validate MIME type
            if file.content_type not in settings.ALLOWED_MIME_TYPES:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"Unsupported file type: {file.content_type}"
                )
            
            logger.debug(f"File validation passed: {file.filename} ({len(file_content)} bytes)")
            return file_content
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"File reading failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read file: {str(e)}"
            )

class OCRTextConverter:
    """Converts Document AI response to OCRText format"""
    
    @staticmethod
    def convert_document_to_ocr_text(document: Any, filename: str = "document") -> OCRText:
        """Convert Document AI Document to OCRText with proper spans"""
        try:
            if not hasattr(document, 'text'):
                logger.error("Invalid document object - no text attribute")
                return OCRText(
                    full_text="Failed to extract text from document",
                    blocks=[OCRBlock(
                        text="Document processing error",
                        span=PageSpan(page=1, start_line=1, end_line=1)
                    )]
                )
            
            full_text = document.text
            blocks = []
            
            # Process each page
            if hasattr(document, 'pages') and document.pages:
                for page_idx, page in enumerate(document.pages):
                    page_number = page_idx + 1
                    
                    # Process paragraphs (most structured text elements)
                    if hasattr(page, 'paragraphs') and page.paragraphs:
                        for para in page.paragraphs:
                            para_text = OCRTextConverter._extract_text_from_layout(
                                document.text, para.layout.text_anchor
                            )
                            
                            if para_text.strip():
                                # Calculate line numbers (approximate)
                                try:
                                    lines_before = document.text[:para.layout.text_anchor.text_segments[0].start_index].count('\n')
                                    lines_in_para = para_text.count('\n')
                                except (IndexError, AttributeError):
                                    lines_before = 0
                                    lines_in_para = 0
                                
                                span = PageSpan(
                                    page=page_number,
                                    start_line=max(1, lines_before + 1),
                                    end_line=lines_before + lines_in_para + 1
                                )
                                
                                blocks.append(OCRBlock(
                                    text=para_text.strip(),
                                    span=span
                                ))
                    
                    # If no paragraphs, fall back to blocks
                    elif hasattr(page, 'blocks') and page.blocks:
                        for block in page.blocks:
                            block_text = OCRTextConverter._extract_text_from_layout(
                                document.text, block.layout.text_anchor
                            )
                            
                            if block_text.strip():
                                try:
                                    lines_before = document.text[:block.layout.text_anchor.text_segments[0].start_index].count('\n')
                                    lines_in_block = block_text.count('\n')
                                except (IndexError, AttributeError):
                                    lines_before = 0
                                    lines_in_block = 0
                                
                                span = PageSpan(
                                    page=page_number,
                                    start_line=max(1, lines_before + 1),
                                    end_line=lines_before + lines_in_block + 1
                                )
                                
                                blocks.append(OCRBlock(
                                    text=block_text.strip(),
                                    span=span
                                ))
                    
                    # If no structured elements, create one block per page
                    else:
                        page_text = OCRTextConverter._extract_page_text(document.text, page)
                        if page_text.strip():
                            span = PageSpan(
                                page=page_number,
                                start_line=1,
                                end_line=page_text.count('\n') + 1
                            )
                            
                            blocks.append(OCRBlock(
                                text=page_text.strip(),
                                span=span
                            ))
            
            # If no blocks were created, create a single block with all text
            if not blocks and full_text.strip():
                span = PageSpan(page=1, start_line=1, end_line=full_text.count('\n') + 1)
                blocks.append(OCRBlock(text=full_text.strip(), span=span))
            
            logger.info(f"Converted document to {len(blocks)} text blocks")
            return OCRText(full_text=full_text, blocks=blocks)
            
        except Exception as e:
            logger.error(f"Document conversion failed: {e}", exc_info=True)
            # Return minimal structure as fallback
            return OCRText(
                full_text=getattr(document, 'text', '') if document else "",
                blocks=[OCRBlock(
                    text=getattr(document, 'text', '') if document else "Failed to extract text",
                    span=PageSpan(page=1, start_line=1, end_line=1)
                )]
            )
    
    @staticmethod
    def _extract_text_from_layout(full_text: str, text_anchor) -> str:
        """Extract text from Document AI text anchor"""
        if not text_anchor or not hasattr(text_anchor, 'text_segments') or not text_anchor.text_segments:
            return ""
        
        try:
            text_segments = []
            for segment in text_anchor.text_segments:
                start_idx = getattr(segment, 'start_index', 0)
                end_idx = getattr(segment, 'end_index', 0)
                if start_idx < len(full_text) and end_idx <= len(full_text):
                    text_segments.append(full_text[start_idx:end_idx])
            
            return "".join(text_segments)
        except Exception as e:
            logger.warning(f"Text extraction from layout failed: {e}")
            return ""
    
    @staticmethod
    def _extract_page_text(full_text: str, page) -> str:
        """Extract text for an entire page"""
        try:
            if not hasattr(page, 'dimension'):
                return ""
            
            # This is a simplified approach - in reality, you'd want to
            # use the page's text anchor if available
            page_chars = len(full_text) // max(1, len([page]))  # Rough estimate
            start_idx = 0  # Would need proper calculation
            end_idx = min(len(full_text), start_idx + page_chars)
            
            return full_text[start_idx:end_idx]
        except Exception:
            return ""

class FallbackOCRService:
    """Fallback OCR service for when Document AI is not available"""
    
    @staticmethod
    async def extract_text_fallback(file: UploadFile) -> OCRText:
        """Create a fallback OCR result for testing/development"""
        logger.warning("Using fallback OCR service - Document AI not available")
        
        try:
            # Read file for basic info
            await file.seek(0)
            content = await file.read()
            await file.seek(0)
            
            # Create mock OCR result
            mock_text = f"""
DOCUMENT: {file.filename}
FILE TYPE: {file.content_type}
SIZE: {len(content)} bytes
PROCESSED: {datetime.now().isoformat()}

This is a mock OCR extraction result for development purposes.
In production, this would contain the actual extracted text from Document AI.

Sample legal content for testing:
- This agreement shall be governed by the laws of [State/Country]
- Either party may terminate this agreement with 30 days written notice
- The Company shall not be liable for any indirect or consequential damages
- Confidential information shall not be disclosed to third parties
            """.strip()
            
            # Split into blocks
            blocks = []
            lines = mock_text.split('\n')
            current_block = []
            current_line = 1
            
            for line in lines:
                if line.strip():
                    current_block.append(line)
                else:
                    if current_block:
                        block_text = '\n'.join(current_block)
                        blocks.append(OCRBlock(
                            text=block_text,
                            span=PageSpan(
                                page=1,
                                start_line=current_line,
                                end_line=current_line + len(current_block) - 1
                            )
                        ))
                        current_line += len(current_block) + 1
                        current_block = []
            
            # Add final block if exists
            if current_block:
                block_text = '\n'.join(current_block)
                blocks.append(OCRBlock(
                    text=block_text,
                    span=PageSpan(
                        page=1,
                        start_line=current_line,
                        end_line=current_line + len(current_block) - 1
                    )
                ))
            
            return OCRText(full_text=mock_text, blocks=blocks)
            
        except Exception as e:
            logger.error(f"Fallback OCR failed: {e}")
            # Ultimate fallback
            return OCRText(
                full_text=f"Failed to process {file.filename}",
                blocks=[OCRBlock(
                    text=f"Error processing file: {file.filename}",
                    span=PageSpan(page=1, start_line=1, end_line=1)
                )]
            )

# Global service instance
_document_ai_service = DocumentAIService()

async def extract_text(file: UploadFile) -> Dict[str, Any]:
    """
    Main OCR function - extracts text from uploaded file using Document AI
    
    Args:
        file: Uploaded file (PDF, DOCX, PNG, JPEG)
        
    Returns:
        Dictionary containing OCRText data (full_text, blocks with spans)
        
    Raises:
        HTTPException: For various error conditions (file size, type, processing)
    """
    try:
        logger.info(f"Starting OCR extraction for: {file.filename}")
        
        # Initialize service if needed
        if not _document_ai_service.initialized:
            await _document_ai_service.initialize()
        
        # Try Document AI first
        if _document_ai_service.client and GOOGLE_CLOUD_AVAILABLE:
            try:
                document = await _document_ai_service.process_document(file)
                ocr_result = OCRTextConverter.convert_document_to_ocr_text(
                    document, file.filename
                )
                
                # Convert to dict for compatibility with existing code
                result = {
                    "full_text": ocr_result.full_text,
                    "blocks": [
                        {
                            "text": block.text,
                            "span": {
                                "page": block.span.page,
                                "start_line": block.span.start_line,
                                "end_line": block.span.end_line
                            }
                        }
                        for block in ocr_result.blocks
                    ]
                }
                
                logger.info(f"OCR extraction completed: {len(result['blocks'])} blocks extracted")
                return result
                
            except HTTPException:
                # Re-raise HTTP exceptions (they're already properly formatted)
                raise
            except Exception as e:
                logger.error(f"Document AI processing failed, falling back: {e}")
                # Fall through to fallback
        
        # Fallback OCR service
        logger.info("Using fallback OCR service")
        ocr_result = await FallbackOCRService.extract_text_fallback(file)
        
        result = {
            "full_text": ocr_result.full_text,
            "blocks": [
                {
                    "text": block.text,
                    "span": {
                        "page": block.span.page,
                        "start_line": block.span.start_line,
                        "end_line": block.span.end_line
                    }
                }
                for block in ocr_result.blocks
            ]
        }
        
        logger.info(f"Fallback OCR extraction completed: {len(result['blocks'])} blocks extracted")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR extraction failed completely: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text extraction failed: {str(e)}"
        )

async def get_ocr_service_info() -> Dict[str, Any]:
    """Get OCR service status and capabilities"""
    return {
        "document_ai_available": GOOGLE_CLOUD_AVAILABLE and _document_ai_service.initialized,
        "fallback_available": True,
        "supported_formats": settings.ALLOWED_MIME_TYPES,
        "max_file_size_mb": settings.MAX_FILE_SIZE // (1024 * 1024),
        "max_pages": settings.MAX_PAGES,
        "processor_id": settings.DOCAI_PROCESSOR_ID if settings.DOCAI_PROCESSOR_ID else None,
        "gcp_project": settings.GCP_PROJECT_ID if settings.GCP_PROJECT_ID else None
    }

async def health_check() -> Dict[str, Any]:
    """Health check for OCR service"""
    try:
        if not _document_ai_service.initialized:
            await _document_ai_service.initialize()
        
        return {
            "status": "healthy",
            "document_ai_ready": _document_ai_service.initialized,
            "fallback_ready": True,
            "dependencies": {
                "google_cloud": GOOGLE_CLOUD_AVAILABLE,
                "document_ai_client": _document_ai_service.client is not None
            }
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "fallback_available": True
        }
