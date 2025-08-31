# src/backend/tests/test_routers/test_analyze.py
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from fastapi import HTTPException, status
from src.backend.main import app
from src.backend.schemas.analysis import Clause, ClauseTag, RiskScore, RiskLevel, PageSpan

# Create test client
client = TestClient(app)

class TestAnalyzeRouter:
    
    def test_analyze_document_success(self, mock_api_key, sample_ocr_text):
        """Test successful document analysis"""
        with patch('src.backend.services.clause.classify_clauses') as mock_classify, \
             patch('src.backend.routers.analyze.score_risks') as mock_risk, \
             patch('src.backend.services.rag.summarize_200w') as mock_summary:
            
            # Mock service responses
            mock_classify.return_value = [
                Clause(
                    id="c-001",
                    tag=ClauseTag.INDEMNITY,
                    text="Test indemnity clause",
                    span=PageSpan(page=1, start_line=1, end_line=1)
                )
            ]
            
            mock_risk.return_value = [
                RiskScore(
                    clause_id="c-001",
                    level=RiskLevel.YELLOW,
                    score=0.4,
                    rationale="Medium risk assessment",
                    supporting_context=[]
                )
            ]
            
            mock_summary.return_value = "This document contains indemnification clauses with medium risk levels."
            
            payload = {
                "ocr": {
                    "full_text": sample_ocr_text["full_text"],
                    "blocks": sample_ocr_text["blocks"]
                },
                "top_k": 5
            }
            
            headers = {
                "x-api-key": mock_api_key,
                "Host": "testserver"
            }
            
            response = client.post("/analyze/", json=payload, headers=headers)
            assert response.status_code == 200
            data = response.json()
            assert "clauses" in data
            assert "risks" in data
            assert "summary_200w" in data

    def test_analyze_document_no_auth(self, sample_ocr_text):
        """Test analyze without authentication"""
        # 🔧 FIX: Temporarily clear dependency override for this test
        from src.backend.main import require_api_key
        
        # Save original override
        original_override = app.dependency_overrides.get(require_api_key)
        
        # Clear the override so authentication works normally
        if require_api_key in app.dependency_overrides:
            del app.dependency_overrides[require_api_key]
        
        try:
            payload = {
                "ocr": {
                    "full_text": sample_ocr_text["full_text"],
                    "blocks": sample_ocr_text["blocks"]
                },
                "top_k": 5
            }
            
            headers = {"Host": "testserver"}  # No x-api-key header
            response = client.post("/analyze/", json=payload, headers=headers)
            assert response.status_code == 401
            
        finally:
            # Restore the original override for other tests
            if original_override:
                app.dependency_overrides[require_api_key] = original_override

    def test_analyze_invalid_payload(self, mock_api_key):
        """Test analyze with invalid payload"""
        payload = {"invalid": "data"}
        
        headers = {
            "x-api-key": mock_api_key,
            "Host": "testserver"
        }
        
        response = client.post("/analyze/", json=payload, headers=headers)
        assert response.status_code == 422  # Validation error

    def test_analyze_service_error(self, mock_api_key, sample_ocr_text):
        """Test analyze with service error"""
        with patch('src.backend.services.clause.classify_clauses') as mock_classify:
            mock_classify.side_effect = Exception("Service error")
            
            payload = {
                "ocr": {
                    "full_text": sample_ocr_text["full_text"],
                    "blocks": sample_ocr_text["blocks"]
                },
                "top_k": 5
            }
            
            headers = {
                "x-api-key": mock_api_key,
                "Host": "testserver"
            }
            
            response = client.post("/analyze/", json=payload, headers=headers)
            assert response.status_code == 500

    def test_analyze_health(self):
        """Test analyze health endpoint"""
        headers = {"Host": "testserver"}
        response = client.get("/analyze/health", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
