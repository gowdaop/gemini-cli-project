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
from contextlib import asynccontextmanager
from functools import lru_cache
import time

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
    from google.api_core.client_options import ClientOptions
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    documentai = None

from fastapi import UploadFile, HTTPException, status
from ..config import settings
from ..schemas.analysis import OCRText, OCRBlock, PageSpan

logger = logging.getLogger(__name__)

class DocumentAIError(Exception):
    """Custom exception for Document AI related errors"""
    pass

class ProcessingTimeoutError(DocumentAIError):
    """Raised when document processing times out"""
    pass

class DocumentAIService:
    """Production-ready Document AI service with comprehensive error handling"""
    
    def __init__(self):
        self.client = None
        self.processor_name = None
        self.initialized = False
        self._max_file_size = settings.MAX_FILE_SIZE
        self._max_pages = settings.MAX_PAGES
        self._retry_config = None
        self._client_options = None
        self._last_health_check = 0
        self._health_check_interval = 300  # 5 minutes
        self._is_healthy = False
        
        if GOOGLE_CLOUD_AVAILABLE:
            self._setup_retry_config()
            self._setup_client_options()
    
    def _setup_retry_config(self):
        """Configure retry policy for robust error handling"""
        self._retry_config = retry.Retry(
            initial=1.0,
            maximum=32.0,  # Increased max backoff
            multiplier=2.0,
            deadline=300.0,  # 5 minutes total deadline
            predicate=retry.if_exception_type(
                gcp_exceptions.DeadlineExceeded,
                gcp_exceptions.ServiceUnavailable,
                gcp_exceptions.InternalServerError,
                gcp_exceptions.TooManyRequests,  # Added rate limiting retry
                gcp_exceptions.Aborted
            )
        )
    
    def _setup_client_options(self):
        """Configure client options for optimal performance"""
        # Use regional endpoint for better performance
        api_endpoint = f"{settings.GCP_LOCATION}-documentai.googleapis.com"
        
        self._client_options = ClientOptions(
            api_endpoint=api_endpoint,
            # Add quota project if different from auth project
            quota_project_id=settings.GCP_PROJECT_ID
        )
    
    async def initialize(self):
        """Initialize Document AI client and processor with improved error handling"""
        if self.initialized:
            return
        
        try:
            if not GOOGLE_CLOUD_AVAILABLE:
                logger.warning("Google Cloud Document AI not available - using fallback mode")
                return
            
            # Validate required settings
            self._validate_settings()
            
            # Initialize client with optimal settings
            self.client = documentai.DocumentProcessorServiceClient(
                client_options=self._client_options
            )
            
            # Build processor name
            self.processor_name = self.client.processor_path(
                settings.GCP_PROJECT_ID,
                settings.GCP_LOCATION,
                settings.DOCAI_PROCESSOR_ID
            )
            
            # Test processor access
            await self._test_processor_access()
            
            self.initialized = True
            self._is_healthy = True
            self._last_health_check = time.time()
            
            logger.info(f"Document AI service initialized successfully")
            logger.info(f"Processor: {self.processor_name}")
            logger.info(f"API Endpoint: {self._client_options.api_endpoint}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Document AI: {e}", exc_info=True)
            # Reset state on failure
            self.initialized = False
            self._is_healthy = False
            # Don't raise exception - allow fallback mode
    
    def _validate_settings(self):
        """Validate all required settings are present"""
        required_settings = [
            ("GCP_PROJECT_ID", settings.GCP_PROJECT_ID),
            ("DOCAI_PROCESSOR_ID", settings.DOCAI_PROCESSOR_ID),
            ("GCP_LOCATION", settings.GCP_LOCATION)
        ]
        
        missing_settings = [name for name, value in required_settings if not value]
        
        if missing_settings:
            raise ValueError(f"Missing required settings: {', '.join(missing_settings)}")
        
        # Validate location format
        if not settings.GCP_LOCATION.replace('-', '').replace('_', '').isalnum():
            raise ValueError(f"Invalid GCP_LOCATION format: {settings.GCP_LOCATION}")
    
    async def _test_processor_access(self):
        """Test processor access with minimal overhead"""
        try:
            # Use get_processor instead of processing a test document
            get_request = documentai.GetProcessorRequest(name=self.processor_name)
            
            processor_info = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.client.get_processor(request=get_request, timeout=30)
            )
            
            logger.debug(f"Processor access test successful: {processor_info.display_name}")
            
        except Exception as e:
            logger.error(f"Document AI processor access test failed: {e}")
            raise DocumentAIError(f"Processor access failed: {e}")
    
    async def process_document(self, file: UploadFile) -> Document:
        """Process document with enhanced error handling and monitoring"""
        if not self.initialized:
            await self.initialize()
        
        if not self.client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Document AI service not available"
            )
        
        # Check service health periodically
        await self._check_service_health()
        
        if not self._is_healthy:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Document AI service is unhealthy"
            )
        
        start_time = time.time()
        
        try:
            # Read and validate file
            file_content = await self._read_and_validate_file(file)
            
            # Create Document AI request with optimized settings
            raw_document = documentai.RawDocument(
                content=file_content,
                mime_type=file.content_type
            )
            
            # Enhanced request configuration
            request = documentai.ProcessRequest(
                name=self.processor_name,
                raw_document=raw_document,
                skip_human_review=True,  # Faster processing
                # Enable field mask for better performance if you only need specific fields
                # field_mask={"paths": ["text", "pages.paragraphs", "pages.tokens"]}
            )
            
            logger.info(f"Processing document: {file.filename} ({len(file_content)} bytes, {file.content_type})")
            
            # Process with enhanced error handling
            response = await self._execute_with_timeout(
                self._process_document_sync,
                args=(request,),
                timeout=180  # 3 minutes timeout
            )
            
            if not response or not response.document:
                raise DocumentAIError("Document processing failed - no document returned")
            
            processing_time = time.time() - start_time
            logger.info(f"Document processed successfully in {processing_time:.2f}s: {len(response.document.pages)} pages")
            
            return response.document
            
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail="Document processing timed out. Try with a smaller document."
            )
        except gcp_exceptions.InvalidArgument as e:
            logger.error(f"Invalid document format: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid document format: {str(e)}"
            )
        except gcp_exceptions.ResourceExhausted as e:
            logger.error(f"Document AI quota exceeded: {e}")
            self._is_healthy = False  # Mark as unhealthy temporarily
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
        except gcp_exceptions.DeadlineExceeded as e:
            logger.error(f"Document AI request timeout: {e}")
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail="Document processing request timed out."
            )
        except Exception as e:
            logger.error(f"Document processing failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Document processing failed: {str(e)}"
            )
    
    def _process_document_sync(self, request):
        """Synchronous document processing with retry logic"""
        return self.client.process_document(
            request=request, 
            retry=self._retry_config,
            timeout=120  # Individual request timeout
        )
    def _validate_settings_comprehensive(self):
        """Comprehensive settings validation with helpful error messages"""
        validation_errors = []
        
        # Required settings check
        required_settings = {
            "GCP_PROJECT_ID": settings.GCP_PROJECT_ID,
            "DOCAI_PROCESSOR_ID": settings.DOCAI_PROCESSOR_ID,
            "GCP_LOCATION": settings.GCP_LOCATION
        }
        
        for name, value in required_settings.items():
            if not value:
                validation_errors.append(f"Missing required setting: {name}")
            elif not isinstance(value, str) or not value.strip():
                validation_errors.append(f"Invalid {name}: must be non-empty string")
        
        # Format validations
        if settings.GCP_PROJECT_ID and not settings.GCP_PROJECT_ID.replace('-', '').replace('_', '').isalnum():
            validation_errors.append("GCP_PROJECT_ID contains invalid characters")
        
        if settings.GCP_LOCATION and not settings.GCP_LOCATION.replace('-', '').replace('_', '').isalnum():
            validation_errors.append("GCP_LOCATION contains invalid characters")
        
        if settings.DOCAI_PROCESSOR_ID and len(settings.DOCAI_PROCESSOR_ID) < 10:
            validation_errors.append("DOCAI_PROCESSOR_ID appears to be too short")
        
        # Size validations
        if settings.MAX_FILE_SIZE and settings.MAX_FILE_SIZE > 100 * 1024 * 1024:  # 100MB
            validation_errors.append("MAX_FILE_SIZE exceeds Document AI limits (100MB)")
        
        if validation_errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {err}" for err in validation_errors)
            raise ValueError(error_msg)
    
    async def _execute_with_timeout(self, func, args=(), kwargs=None, timeout=180):
        """Execute function with timeout in thread pool"""
        if kwargs is None:
            kwargs = {}
        
        try:
            return await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, func, *args, **kwargs),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Operation timed out after {timeout} seconds")
            raise ProcessingTimeoutError(f"Operation timed out after {timeout} seconds")
    
    async def _check_service_health(self):
        """Periodic health check to ensure service is working"""
        current_time = time.time()
        
        if current_time - self._last_health_check < self._health_check_interval:
            return
        
        try:
            # Quick processor info check
            get_request = documentai.GetProcessorRequest(name=self.processor_name)
            await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.client.get_processor(request=get_request, timeout=10)
            )
            
            self._is_healthy = True
            self._last_health_check = current_time
            logger.debug("Document AI health check passed")
            
        except Exception as e:
            logger.warning(f"Document AI health check failed: {e}")
            self._is_healthy = False
    
    async def _read_and_validate_file(self, file: UploadFile) -> bytes:
        """Enhanced file validation with security checks"""
        try:
            # Reset file pointer
            await file.seek(0)
            file_content = await file.read()
            await file.seek(0)
            
            # Enhanced validation
            self._validate_file_content(file, file_content)
            
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
    
    def _validate_file_content(self, file: UploadFile, content: bytes):
        """Comprehensive file validation"""
        # Basic size checks
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded"
            )
        
        if len(content) > self._max_file_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size ({len(content)} bytes) exceeds maximum ({self._max_file_size} bytes)"
            )
        
        # MIME type validation
        if file.content_type not in settings.ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type: {file.content_type}. Allowed types: {', '.join(settings.ALLOWED_MIME_TYPES)}"
            )
        
        # Additional security checks
        self._security_scan_content(content, file.content_type)
    
    def _security_scan_content(self, content: bytes, mime_type: str):
        """Basic security scanning of file content"""
        # Check for common malicious patterns
        malicious_patterns = [
            b'<script',
            b'javascript:',
            b'<?php',
            b'<%',
        ]
        
        content_lower = content[:1024].lower()  # Check first 1KB
        for pattern in malicious_patterns:
            if pattern in content_lower:
                logger.warning(f"Potentially malicious pattern detected: {pattern}")
                # Could raise exception or just log warning
        
        # Validate PDF header if PDF
        if mime_type == 'application/pdf' and not content.startswith(b'%PDF'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid PDF file format"
            )

class EnhancedOCRTextConverter:
    """Enhanced converter with better text extraction and structure preservation"""
    
    @staticmethod
    def convert_document_to_ocr_text(document: Document, filename: str = "document") -> OCRText:
        """Enhanced document conversion with better structure detection"""
        try:
            if not hasattr(document, 'text'):
                logger.error("Invalid document object - no text attribute")
                return EnhancedOCRTextConverter._create_error_result(
                    "Failed to extract text from document"
                )
            
            full_text = document.text
            blocks = []
            
            # Try different extraction strategies based on document structure
            if hasattr(document, 'pages') and document.pages:
                blocks = EnhancedOCRTextConverter._extract_structured_blocks(document, full_text)
            
            # Fallback to simple text blocks if no structured content
            if not blocks and full_text.strip():
                blocks = EnhancedOCRTextConverter._create_simple_blocks(full_text)
            
            # Post-process blocks for better quality
            blocks = EnhancedOCRTextConverter._post_process_blocks(blocks)
            
            logger.info(f"Converted document to {len(blocks)} text blocks")
            return OCRText(full_text=full_text, blocks=blocks)
            
        except Exception as e:
            logger.error(f"Document conversion failed: {e}", exc_info=True)
            return EnhancedOCRTextConverter._create_error_result(
                getattr(document, 'text', 'Failed to extract text') if document else "No document provided"
            )
    
    @staticmethod
    def _extract_structured_blocks(document: Document, full_text: str) -> List[OCRBlock]:
        """Extract blocks using document structure information"""
        blocks = []
        
        for page_idx, page in enumerate(document.pages):
            page_number = page_idx + 1
            
            # Try paragraphs first (most structured)
            if hasattr(page, 'paragraphs') and page.paragraphs:
                for para in page.paragraphs:
                    block = EnhancedOCRTextConverter._create_block_from_layout(
                        full_text, para.layout, page_number
                    )
                    if block:
                        blocks.append(block)
                        
            # Fallback to lines if no paragraphs
            elif hasattr(page, 'lines') and page.lines:
                for line in page.lines:
                    block = EnhancedOCRTextConverter._create_block_from_layout(
                        full_text, line.layout, page_number
                    )
                    if block:
                        blocks.append(block)
            
            # Ultimate fallback to tokens
            elif hasattr(page, 'tokens') and page.tokens:
                # Group tokens into meaningful blocks
                token_groups = EnhancedOCRTextConverter._group_tokens(page.tokens, full_text)
                for token_group in token_groups:
                    block = EnhancedOCRTextConverter._create_block_from_tokens(
                        token_group, full_text, page_number
                    )
                    if block:
                        blocks.append(block)
        
        return blocks
    
    @staticmethod
    def _create_block_from_layout(full_text: str, layout, page_number: int) -> Optional[OCRBlock]:
        """Create OCRBlock from Document AI layout element"""
        try:
            text = EnhancedOCRTextConverter._extract_text_from_layout(full_text, layout.text_anchor)
            if not text.strip():
                return None
            
            # Calculate line spans more accurately
            start_pos = layout.text_anchor.text_segments[0].start_index if layout.text_anchor.text_segments else 0
            lines_before = full_text[:start_pos].count('\n')
            lines_in_text = text.count('\n')
            
            span = PageSpan(
                page=page_number,
                start_line=max(1, lines_before + 1),
                end_line=lines_before + lines_in_text + 1
            )
            
            return OCRBlock(text=text.strip(), span=span)
            
        except Exception as e:
            logger.warning(f"Failed to create block from layout: {e}")
            return None
    
    @staticmethod
    def _create_simple_blocks(full_text: str) -> List[OCRBlock]:
        """Create simple blocks by splitting on double newlines"""
        blocks = []
        paragraphs = full_text.split('\n\n')
        current_line = 1
        
        for para in paragraphs:
            para = para.strip()
            if para:
                lines_in_para = para.count('\n') + 1
                span = PageSpan(
                    page=1,
                    start_line=current_line,
                    end_line=current_line + lines_in_para - 1
                )
                blocks.append(OCRBlock(text=para, span=span))
                current_line += lines_in_para + 1  # +1 for the separator
        
        return blocks
    
    @staticmethod
    def _post_process_blocks(blocks: List[OCRBlock]) -> List[OCRBlock]:
        """Post-process blocks to improve quality"""
        processed_blocks = []
        
        for block in blocks:
            # Skip very short blocks that are likely noise
            if len(block.text.strip()) < 3:
                continue
            
            # Clean up text
            cleaned_text = EnhancedOCRTextConverter._clean_text(block.text)
            if cleaned_text:
                processed_blocks.append(OCRBlock(
                    text=cleaned_text,
                    span=block.span
                ))
        
        return processed_blocks
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        lines = []
        for line in text.split('\n'):
            cleaned_line = ' '.join(line.split())
            if cleaned_line:
                lines.append(cleaned_line)
        
        return '\n'.join(lines)
    
    @staticmethod
    def _create_error_result(error_text: str) -> OCRText:
        """Create error result with proper structure"""
        return OCRText(
            full_text=error_text,
            blocks=[OCRBlock(
                text=error_text,
                span=PageSpan(page=1, start_line=1, end_line=1)
            )]
        )
    
    @staticmethod
    def _extract_text_from_layout(full_text: str, text_anchor) -> str:
        """Enhanced text extraction from Document AI text anchor"""
        if not text_anchor or not hasattr(text_anchor, 'text_segments') or not text_anchor.text_segments:
            return ""
        
        try:
            text_segments = []
            for segment in text_anchor.text_segments:
                start_idx = getattr(segment, 'start_index', 0)
                end_idx = getattr(segment, 'end_index', 0)
                if 0 <= start_idx < len(full_text) and start_idx < end_idx <= len(full_text):
                    text_segments.append(full_text[start_idx:end_idx])
            
            return "".join(text_segments)
        except Exception as e:
            logger.warning(f"Text extraction from layout failed: {e}")
            return ""
    
    @staticmethod
    def _group_tokens(tokens, full_text: str) -> List[List]:
        """Group tokens into logical blocks (lines/paragraphs)"""
        # This is a simplified implementation - could be enhanced
        # based on token positions and spacing
        return [tokens]  # Simple grouping for now
    @staticmethod
    def _create_simple_blocks_from_text(text: str) -> OCRText:
        """Create OCRText from simple text string"""
        blocks = EnhancedOCRTextConverter._create_simple_blocks(text)
        return OCRText(full_text=text, blocks=blocks)
    
    @staticmethod
    def _create_block_from_tokens(token_group, full_text: str, page_number: int) -> Optional[OCRBlock]:
        """Create block from token group"""
        # Implementation would extract text from token positions
        # This is a placeholder for the complex logic needed
        return None
    @staticmethod
    def _group_tokens(tokens, full_text: str) -> List[List]:
        """Group tokens into logical blocks based on position and spacing"""
        if not tokens:
            return []
        
        groups = []
        current_group = []
        last_y_position = None
        
        for token in tokens:
            try:
                # Get bounding box if available
                if hasattr(token, 'layout') and hasattr(token.layout, 'bounding_poly'):
                    vertices = token.layout.bounding_poly.vertices
                    if vertices:
                        current_y = vertices[0].y
                        
                        # Start new group if significant vertical gap
                        if last_y_position is not None and abs(current_y - last_y_position) > 20:
                            if current_group:
                                groups.append(current_group)
                                current_group = []
                        
                        current_group.append(token)
                        last_y_position = current_y
                    else:
                        current_group.append(token)
                else:
                    current_group.append(token)
                    
            except Exception as e:
                logger.warning(f"Error processing token: {e}")
                current_group.append(token)
        
        if current_group:
            groups.append(current_group)
        
        return groups or [tokens]  # Fallback to all tokens in one group

    @staticmethod
    def _create_block_from_tokens(token_group, full_text: str, page_number: int) -> Optional[OCRBlock]:
        """Create block from token group with proper text extraction"""
        if not token_group:
            return None
        
        try:
            # Extract text from all tokens in group
            text_parts = []
            start_positions = []
            
            for token in token_group:
                if hasattr(token, 'layout') and hasattr(token.layout, 'text_anchor'):
                    token_text = EnhancedOCRTextConverter._extract_text_from_layout(
                        full_text, token.layout.text_anchor
                    )
                    if token_text.strip():
                        text_parts.append(token_text)
                        # Track position for line calculation
                        if token.layout.text_anchor.text_segments:
                            start_positions.append(token.layout.text_anchor.text_segments[0].start_index)
            
            if not text_parts:
                return None
            
            # Combine text parts
            combined_text = ' '.join(text_parts).strip()
            
            # Calculate line spans
            if start_positions:
                min_pos = min(start_positions)
                lines_before = full_text[:min_pos].count('\n')
                lines_in_text = combined_text.count('\n')
                
                span = PageSpan(
                    page=page_number,
                    start_line=max(1, lines_before + 1),
                    end_line=lines_before + lines_in_text + 1
                )
            else:
                span = PageSpan(page=page_number, start_line=1, end_line=1)
            
            return OCRBlock(text=combined_text, span=span)
            
        except Exception as e:
            logger.warning(f"Failed to create block from tokens: {e}")
            return None

# Rest of your existing code (FallbackOCRService, etc.) remains the same
# but update the converter reference:
class ServiceMetrics:
    """Track service performance metrics"""
    
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_processing_time = 0.0
        self.average_processing_time = 0.0
        self.last_reset = time.time()
    
    def record_request(self, success: bool, processing_time: float):
        """Record request metrics"""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.total_processing_time += processing_time
        self.average_processing_time = self.total_processing_time / self.total_requests
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        uptime = time.time() - self.last_reset
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0,
            "average_processing_time": round(self.average_processing_time, 2),
            "uptime_seconds": round(uptime, 2)
        }
    
    def reset(self):
        """Reset all metrics"""
        self.__init__()

# Add to DocumentAIService class
def __init__(self):
    # ... existing initialization ...
    self._metrics = ServiceMetrics()

# Update process_document method to record metrics
async def process_document(self, file: UploadFile) -> Document:
    start_time = time.time()
    success = False
    
    try:
        # ... existing processing logic ...
        # Read and validate file
        file_content = await self._read_and_validate_file(file)

        # Create Document AI request with optimized settings
        raw_document = documentai.RawDocument(
            content=file_content,
            mime_type=file.content_type
        )

        request = documentai.ProcessRequest(
            name=self.processor_name,
            raw_document=raw_document,
            skip_human_review=True,
        )

        document = await self._execute_with_timeout(
            self._process_document_sync,
            args=(request,),
            timeout=180  # 3 minutes timeout
        )
        result = document
        success = True
        return result
    except Exception as e:
        success = False
        raise
    finally:
        processing_time = time.time() - start_time
        self._metrics.record_request(success, processing_time)

class FallbackOCRService:
    """Enhanced fallback OCR service"""
    
    @staticmethod
    async def extract_text_fallback(file: UploadFile) -> OCRText:
        """Enhanced fallback OCR result"""
        logger.warning("Using fallback OCR service - Document AI not available")
        
        try:
            await file.seek(0)
            content = await file.read()
            await file.seek(0)
            
            # Create more realistic mock content based on file type
            if file.content_type == 'application/pdf':
                mock_text = FallbackOCRService._generate_pdf_mock(file.filename, len(content))
            elif file.content_type.startswith('image/'):
                mock_text = FallbackOCRService._generate_image_mock(file.filename, len(content))
            else:
                mock_text = FallbackOCRService._generate_generic_mock(file.filename, len(content))
            
            return EnhancedOCRTextConverter._create_simple_blocks_from_text(mock_text)
            
        except Exception as e:
            logger.error(f"Fallback OCR failed: {e}")
            return EnhancedOCRTextConverter._create_error_result(f"Failed to process {file.filename}")
    
    @staticmethod
    def _generate_pdf_mock(filename: str, size: int) -> str:
        """Generate realistic PDF mock content"""
        return f"""CONTRACT AGREEMENT

This Agreement is entered into on {datetime.now().strftime('%B %d, %Y')}.

PARTIES:
Party A: [Company Name]
Party B: [Contractor Name]

TERMS AND CONDITIONS:

1. SCOPE OF WORK
   The Contractor agrees to provide services as specified in Exhibit A.

2. PAYMENT TERMS
   Payment shall be made within 30 days of invoice receipt.

3. CONFIDENTIALITY
   Both parties agree to maintain confidentiality of proprietary information.

4. TERMINATION
   Either party may terminate this agreement with 30 days written notice.

5. GOVERNING LAW
   This agreement shall be governed by the laws of [Jurisdiction].

[Document: {filename}, Size: {size} bytes]
[Processed with fallback OCR service for development purposes]"""
    
    @staticmethod
    def _generate_image_mock(filename: str, size: int) -> str:
        """Generate image mock content"""
        return f"""IMAGE DOCUMENT EXTRACT

Extracted text from image: {filename}

This appears to be a scanned document or image file.
In production, Document AI would extract the actual text content.

File Information:
- Filename: {filename}
- Size: {size} bytes
- Type: Image document

Sample extracted content:
[This would contain the actual OCR results from the image]
"""
    
    @staticmethod
    def _generate_generic_mock(filename: str, size: int) -> str:
        """Generate generic mock content"""
        return f"""DOCUMENT ANALYSIS

File: {filename}
Size: {size} bytes
Processed: {datetime.now().isoformat()}

This is a development mode extraction.
In production, Document AI would provide actual content analysis.
"""

# Update the global service instance and main functions
_document_ai_service = DocumentAIService()

async def extract_text(file: UploadFile) -> Dict[str, Any]:
    """Enhanced main OCR function with better error handling"""
    try:
        logger.info(f"Starting OCR extraction for: {file.filename}")
        
        # Initialize service if needed
        if not _document_ai_service.initialized:
            await _document_ai_service.initialize()
        
        # Try Document AI first
        if _document_ai_service.client and GOOGLE_CLOUD_AVAILABLE:
            try:
                document = await _document_ai_service.process_document(file)
                ocr_result = EnhancedOCRTextConverter.convert_document_to_ocr_text(
                    document, file.filename
                )
                
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
                    ],
                    "metadata": {
                        "processor_used": "document_ai",
                        "pages_processed": len(document.pages) if hasattr(document, 'pages') and document.pages else 1,
                        "processing_time": time.time()
                    }
                }
                
                logger.info(f"OCR extraction completed: {len(result['blocks'])} blocks extracted")
                return result
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Document AI processing failed, falling back: {e}")
        
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
            ],
            "metadata": {
                "processor_used": "fallback",
                "processing_time": time.time()
            }
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

# Enhanced service info and health check functions
async def get_ocr_service_info() -> Dict[str, Any]:
    """Enhanced OCR service status and capabilities"""
    return {
        "document_ai_available": GOOGLE_CLOUD_AVAILABLE and _document_ai_service.initialized,
        "document_ai_healthy": _document_ai_service._is_healthy,
        "fallback_available": True,
        "supported_formats": list(settings.ALLOWED_MIME_TYPES),
        "max_file_size_mb": settings.MAX_FILE_SIZE // (1024 * 1024),
        "max_pages": settings.MAX_PAGES,
        "processor_info": {
            "processor_id": settings.DOCAI_PROCESSOR_ID if settings.DOCAI_PROCESSOR_ID else None,
            "gcp_project": settings.GCP_PROJECT_ID if settings.GCP_PROJECT_ID else None,
            "location": settings.GCP_LOCATION if settings.GCP_LOCATION else None,
            "endpoint": getattr(_document_ai_service._client_options, 'api_endpoint', None) if _document_ai_service._client_options else None
        },
        "service_stats": {
            "last_health_check": _document_ai_service._last_health_check,
            "health_check_interval": _document_ai_service._health_check_interval
        }
    }

async def health_check() -> Dict[str, Any]:
    """Enhanced health check with detailed diagnostics"""
    try:
        if not _document_ai_service.initialized:
            await _document_ai_service.initialize()
        
        # Perform health check
        await _document_ai_service._check_service_health()
        
        return {
            "status": "healthy" if _document_ai_service._is_healthy else "degraded",
            "document_ai_ready": _document_ai_service.initialized,
            "document_ai_healthy": _document_ai_service._is_healthy,
            "fallback_ready": True,
            "dependencies": {
                "google_cloud_available": GOOGLE_CLOUD_AVAILABLE,
                "document_ai_client_initialized": _document_ai_service.client is not None,
                "processor_accessible": _document_ai_service.processor_name is not None
            },
            "last_health_check": _document_ai_service._last_health_check,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "fallback_available": True,
            "timestamp": time.time()
        }
