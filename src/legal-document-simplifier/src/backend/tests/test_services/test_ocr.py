import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.backend.services import ocr
from src.backend.schemas.analysis import OCRText, OCRBlock, PageSpan

@pytest.fixture
def mock_file():
    """Mock file upload for testing"""
    mock = MagicMock()
    mock.filename = "test.pdf"
    mock.content_type = "application/pdf" 
    mock.read = AsyncMock(return_value=b"fake pdf content")
    mock.seek = AsyncMock(return_value=None)
    return mock

class TestDocumentAIService:
    
    @pytest.fixture
    def doc_ai_service(self):
        return ocr.DocumentAIService()
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, doc_ai_service):
        """Test successful DocumentAI initialization"""
        with patch('src.backend.services.ocr.GOOGLE_CLOUD_AVAILABLE', True), \
             patch('src.backend.services.ocr.documentai') as mock_documentai, \
             patch.object(doc_ai_service, '_test_processor_access', new_callable=AsyncMock):
            
            mock_client = MagicMock()
            mock_documentai.DocumentProcessorServiceClient.return_value = mock_client
            mock_client.processor_path.return_value = "test/processor/path"
            
            await doc_ai_service.initialize()
            
            assert doc_ai_service.initialized
            assert doc_ai_service.client is not None

    @pytest.mark.asyncio
    async def test_initialize_unavailable(self, doc_ai_service):
        """Test initialization when Google Cloud unavailable"""
        with patch('src.backend.services.ocr.GOOGLE_CLOUD_AVAILABLE', False):
            await doc_ai_service.initialize()
            
            assert not doc_ai_service.initialized
            assert doc_ai_service.client is None

    @pytest.mark.asyncio
    async def test_process_document_success(self, doc_ai_service, mock_file):
        """Test successful document processing - FIXED VERSION"""
        doc_ai_service.initialized = True
        doc_ai_service.client = MagicMock()
        doc_ai_service.processor_name = "test/processor"
        
        # Mock DocumentAI response
        mock_document = MagicMock()
        mock_document.text = "Sample document text"
        mock_document.pages = []
        mock_response = MagicMock()
        mock_response.document = mock_document
        
        # CRITICAL FIX: Make client.process_request return mock_response synchronously
        # This is called inside run_in_executor, so it must be sync, not async
        doc_ai_service.client.process_request.return_value = mock_response
        
        # Mock the file reading method
        with patch.object(doc_ai_service, '_read_and_validate_file', new_callable=AsyncMock) as mock_read:
            mock_read.return_value = b"fake content"
            
            # Call the method - no need to patch run_in_executor anymore
            result = await doc_ai_service.process_document(mock_file)
            
            # Verify the result
            assert result == mock_document
            mock_read.assert_called_once_with(mock_file)
            doc_ai_service.client.process_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_document_not_initialized(self, doc_ai_service, mock_file):
        """Test document processing when not initialized"""
        doc_ai_service.initialized = False
        doc_ai_service.client = None
        
        with pytest.raises(Exception):
            await doc_ai_service.process_document(mock_file)

class TestOCRTextConverter:
    
    def test_convert_document_to_ocr_text(self):
        """Test document conversion to OCR text"""
        # Mock DocumentAI document
        mock_document = MagicMock()
        mock_document.text = "Sample legal document text"
        mock_document.pages = []
        
        result = ocr.OCRTextConverter.convert_document_to_ocr_text(mock_document)
        
        assert isinstance(result, OCRText)
        assert result.full_text == "Sample legal document text"

    def test_convert_document_invalid(self):
        """Test conversion with invalid document"""
        result = ocr.OCRTextConverter.convert_document_to_ocr_text(None)
        
        assert isinstance(result, OCRText)
        assert len(result.blocks) > 0

class TestFallbackOCRService:
    
    @pytest.mark.asyncio
    async def test_extract_text_fallback(self, mock_file):
        """Test fallback OCR extraction"""
        result = await ocr.FallbackOCRService.extract_text_fallback(mock_file)
        
        assert isinstance(result, OCRText)
        assert len(result.full_text) > 0
        assert len(result.blocks) > 0
        assert "mock" in result.full_text.lower() or "development" in result.full_text.lower()

class TestOCRServiceAPI:
    
    @pytest.mark.asyncio
    async def test_extract_text_success(self, mock_file):
        """Test successful text extraction"""
        with patch('src.backend.services.ocr._document_ai_service') as mock_service:
            mock_service.initialized = True
            mock_service.client = MagicMock()
            
            mock_document = MagicMock()
            mock_document.text = "Extracted text"
            mock_service.process_document = AsyncMock(return_value=mock_document)
            
            with patch.object(ocr.OCRTextConverter, 'convert_document_to_ocr_text') as mock_convert:
                mock_convert.return_value = OCRText(
                    full_text="Extracted text",
                    blocks=[]
                )
                
                result = await ocr.extract_text(mock_file)
                
                assert "full_text" in result
                assert result["full_text"] == "Extracted text"

    @pytest.mark.asyncio
    async def test_extract_text_fallback(self, mock_file):
        """Test text extraction fallback"""
        with patch('src.backend.services.ocr._document_ai_service') as mock_service:
            mock_service.initialized = False
            mock_service.initialize = AsyncMock()
            mock_service.client = None
            
            result = await ocr.extract_text(mock_file)
            
            assert "full_text" in result
            assert len(result["full_text"]) > 0

    @pytest.mark.asyncio
    async def test_get_ocr_service_info(self):
        """Test OCR service info"""
        result = await ocr.get_ocr_service_info()
        
        assert "document_ai_available" in result
        assert "fallback_available" in result
        assert "supported_formats" in result

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test OCR service health check"""
        result = await ocr.health_check()
        
        assert "status" in result
        assert result["status"] in ["healthy", "degraded"]
