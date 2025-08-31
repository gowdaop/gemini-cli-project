# src/backend/tests/test_routers/test_chat.py
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.backend.main import app

# Use synchronous TestClient
client = TestClient(app)

class TestChatRouter:
    
    def test_chat_success(self, mock_api_key):
        """Test successful chat interaction - FINAL FIX: Use integers for chunk_id"""
        
        # 🎯 FINAL FIX: Use integers for chunk_id instead of strings
        sample_rag_contexts = [
            {
                "chunk_id": 1,  # ✅ INTEGER instead of "test-chunk-1"
                "content": "Indemnification clauses protect parties from legal claims and financial losses.",
                "doc_type": "legal_document",
                "jurisdiction": "US",
                "date": "2024-01-01",
                "source_url": "https://example.com/legal-doc",
                "similarity": 0.85
            },
            {
                "chunk_id": 2,  # ✅ INTEGER instead of "test-chunk-2"
                "content": "These clauses typically specify which party bears responsibility for damages.",
                "doc_type": "legal_document",
                "jurisdiction": "US",
                "date": "2024-01-01",
                "source_url": "https://example.com/legal-doc-2",
                "similarity": 0.78
            }
        ]
        
        # Mock the retrieve_comprehensive_evidence function directly
        with patch('src.backend.routers.chat.retrieve_comprehensive_evidence') as mock_retrieve, \
             patch('src.backend.routers.chat.ResponseGenerator.generate_legal_response') as mock_response:
            
            mock_retrieve.return_value = sample_rag_contexts
            mock_response.return_value = "This is an indemnification clause that protects one party from claims."
            
            headers = {"x-api-key": mock_api_key, "Host": "testserver"}
            payload = {
                "question": "What is an indemnification clause?",
                "conversation_id": None
            }
            
            response = client.post("/chat/", json=payload, headers=headers)
            
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "evidence" in data
            assert "conversation_id" in data
            assert len(data["evidence"]) > 0  # This will now pass!

    def test_chat_with_conversation_id(self, mock_api_key):
        """Test chat with existing conversation"""
        with patch('src.backend.routers.chat.retrieve_comprehensive_evidence') as mock_retrieve, \
             patch('src.backend.routers.chat.ResponseGenerator.generate_legal_response') as mock_response:
            
            mock_retrieve.return_value = []
            mock_response.return_value = "Test response"
            
            headers = {"x-api-key": mock_api_key, "Host": "testserver"}
            payload = {
                "question": "Follow up question",
                "conversation_id": "test-conversation-123"
            }
            
            response = client.post("/chat/", json=payload, headers=headers)
            assert response.status_code == 200
            data = response.json()
            assert "conversation_id" in data

    def test_chat_empty_question(self, mock_api_key):
        """Test chat with empty question"""
        headers = {"x-api-key": mock_api_key, "Host": "testserver"}
        payload = {"question": ""}
        
        response = client.post("/chat/", json=payload, headers=headers)
        assert response.status_code in [400, 422]

    def test_chat_too_long_question(self, mock_api_key):
        """Test chat with overly long question"""
        headers = {"x-api-key": mock_api_key, "Host": "testserver"}
        payload = {"question": "x" * 1001}
        
        response = client.post("/chat/", json=payload, headers=headers)
        assert response.status_code == 400

    def test_get_conversation_history_success(self, mock_api_key):
        """Test retrieving conversation history"""
        with patch('src.backend.routers.chat.retrieve_comprehensive_evidence') as mock_retrieve, \
             patch('src.backend.routers.chat.ResponseGenerator.generate_legal_response') as mock_response:
            
            mock_retrieve.return_value = []
            mock_response.return_value = "Test response"
            
            headers = {"x-api-key": mock_api_key, "Host": "testserver"}
            create_payload = {"question": "Test question"}
            create_response = client.post("/chat/", json=create_payload, headers=headers)
            conversation_id = create_response.json()["conversation_id"]
            
            response = client.get(f"/chat/conversations/{conversation_id}", headers=headers)
            assert response.status_code == 200
            data = response.json()
            assert "conversation_id" in data
            assert "history" in data

    def test_get_conversation_history_not_found(self, mock_api_key):
        """Test retrieving non-existent conversation"""
        headers = {"x-api-key": mock_api_key, "Host": "testserver"}
        response = client.get("/chat/conversations/non-existent", headers=headers)
        assert response.status_code == 404

    def test_delete_conversation_success(self, mock_api_key):
        """Test deleting conversation"""
        with patch('src.backend.routers.chat.retrieve_comprehensive_evidence') as mock_retrieve, \
             patch('src.backend.routers.chat.ResponseGenerator.generate_legal_response') as mock_response:
            
            mock_retrieve.return_value = []
            mock_response.return_value = "Test response"
            
            headers = {"x-api-key": mock_api_key, "Host": "testserver"}
            create_payload = {"question": "Test question"}
            create_response = client.post("/chat/", json=create_payload, headers=headers)
            conversation_id = create_response.json()["conversation_id"]
            
            response = client.delete(f"/chat/conversations/{conversation_id}", headers=headers)
            assert response.status_code == 200
            data = response.json()
            assert "message" in data

    def test_chat_health(self, mock_api_key):
        """Test chat health endpoint"""
        headers = {"x-api-key": mock_api_key, "Host": "testserver"}
        response = client.get("/chat/health", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_chat_stats(self, mock_api_key):
        """Test chat statistics endpoint"""
        headers = {"x-api-key": mock_api_key, "Host": "testserver"}
        response = client.get("/chat/stats", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "total_conversations" in data
        assert "total_messages" in data
