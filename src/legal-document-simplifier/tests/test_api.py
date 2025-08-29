import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.backend.main import app
from src.backend.config import settings


class TestAPI:
    """Test suite for API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    @pytest.fixture
    def valid_headers(self):
        """Valid API key headers"""
        return {"x-api-key": settings.API_KEY}
    
    def test_health_check_no_auth(self, client):
        """Test health check endpoint without authentication"""
        response = client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "timestamp" in data
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == settings.API_TITLE
        assert data["version"] == settings.API_VERSION
    
    def test_unauthorized_without_api_key(self, client):
        """Test that endpoints require API key"""
        endpoints = ["/upload", "/analyze", "/chat"]
        
        for endpoint in endpoints:
            response = client.post(endpoint)
            assert response.status_code == 401
            assert "API key" in response.json()["detail"]
    
    def test_unauthorized_with_invalid_api_key(self, client):
        """Test invalid API key rejection"""
        headers = {"x-api-key": "invalid-key"}
        endpoints = ["/upload", "/analyze", "/chat"]
        
        for endpoint in endpoints:
            response = client.post(endpoint, headers=headers)
            assert response.status_code == 401
            assert "Invalid API key" in response.json()["detail"]
    
    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options("/", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST"
        })
        
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
    
    @patch('src.backend.services.ocr.extract_text')
    def test_upload_endpoint_authorized(self, mock_ocr, client, valid_headers):
        """Test upload endpoint with valid authentication"""
        # Mock OCR response
        from src.backend.schemas.analysis import OCRText, OCRBlock, PageSpan
        
        mock_ocr_text = OCRText(
            full_text="Sample contract text",
            blocks=[
                OCRBlock(
                    text="Sample contract text",
                    span=PageSpan(page=1, start_line=1, end_line=1)
                )
            ]
        )
        mock_ocr.return_value = mock_ocr_text
        
        # Test file upload
        test_file = ("test.pdf", b"fake pdf content", "application/pdf")
        response = client.post(
            "/upload",
            files={"file": test_file},
            headers=valid_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "ocr" in data
        assert data["ocr"]["full_text"] == "Sample contract text"
    
    def test_upload_unsupported_file_type(self, client, valid_headers):
        """Test upload with unsupported file type"""
        test_file = ("test.txt", b"text content", "text/plain")
        response = client.post(
            "/upload",
            files={"file": test_file},
            headers=valid_headers
        )
        
        assert response.status_code == 415
        assert "Unsupported file type" in response.json()["detail"]
    
    @patch('src.backend.services.rag.retrieve_contexts')
    @patch('src.backend.services.rag.answer_with_vertex')
    def test_chat_endpoint_authorized(self, mock_answer, mock_retrieve, client, valid_headers):
        """Test chat endpoint with valid authentication"""
        from src.backend.schemas.analysis import RAGContextItem
        
        # Mock dependencies
        mock_retrieve.return_value = [
            RAGContextItem(
                chunk_id=1,
                content="Sample legal context",
                doc_type="contract",
                jurisdiction="US",
                date="2024-01-01",
                source_url="http://example.com",
                similarity=0.95
            )
        ]
        mock_answer.return_value = "This is a sample answer"
        
        chat_data = {
            "question": "What is the liability clause?",
            "conversation_id": None,
            "ocr": None,
            "summary_hint": None
        }
        
        response = client.post(
            "/chat",
            json=chat_data,
            headers=valid_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "evidence" in data
        assert "conversation_id" in data
        assert len(data["evidence"]) == 1
    
    def test_openapi_schema_contains_required_tags(self, client):
        """Test OpenAPI schema contains required tags"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_spec = response.json()
        
        # Extract all tags from paths
        all_tags = set()
        for path_data in openapi_spec["paths"].values():
            for method_data in path_data.values():
                if isinstance(method_data, dict) and "tags" in method_data:
                    all_tags.update(method_data["tags"])
        
        # Check required tags are present
        required_tags = {"upload", "analyze", "chat"}
        assert required_tags.issubset(all_tags)
        
        # Check security scheme is defined
        assert "securitySchemes" in openapi_spec["components"]
        assert "ApiKeyAuth" in openapi_spec["components"]["securitySchemes"]
    
    def test_request_timing_header(self, client):
        """Test that timing header is added to responses"""
        response = client.get("/healthz")
        assert "X-Process-Time" in response.headers
        
        # Timing should be a valid float
        timing = float(response.headers["X-Process-Time"])
        assert timing >= 0.0
    
    @pytest.mark.parametrize("endpoint", ["/upload", "/analyze", "/chat"])
    def test_all_protected_endpoints_require_auth(self, client, endpoint):
        """Parametrized test for all protected endpoints"""
        response = client.post(endpoint)
        assert response.status_code == 401
    
    def test_error_response_format(self, client):
        """Test error responses follow ErrorResponse schema"""
        response = client.post("/upload")  # No auth
        
        assert response.status_code == 401
        error_data = response.json()
        assert "detail" in error_data
        assert isinstance(error_data["detail"], str)


class TestConfiguration:
    """Test configuration and settings"""
    
    def test_settings_validation(self):
        """Test settings are properly configured"""
        assert settings.API_TITLE
        assert settings.API_VERSION
        assert settings.API_KEY
        assert isinstance(settings.CORS_ORIGINS, list)
        assert settings.MAX_FILE_SIZE > 0
        assert len(settings.ALLOWED_MIME_TYPES) > 0
    
    def test_cors_origins_parsing(self):
        """Test CORS origins can be parsed from string"""
        # This would test the validator in actual usage
        assert "http://localhost:3000" in settings.CORS_ORIGINS


if __name__ == "__main__":
    pytest.main([__file__])
