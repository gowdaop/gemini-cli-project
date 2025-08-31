import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from typing import List

from src.backend.services import rag
from src.backend.schemas.analysis import RAGContextItem


class TestRAGService:
    
    @pytest.fixture
    def sample_rag_contexts(self) -> List[RAGContextItem]:
        """Create sample RAG context items for testing"""
        return [
            RAGContextItem(
                chunk_id=123,
                content="Sample legal precedent about indemnification clauses in commercial contracts.",
                doc_type="legal_document",
                jurisdiction="US",
                date="2023-01-01",
                source_url="test://sample1.pdf",
                similarity=0.95
            ),
            RAGContextItem(
                chunk_id=124,
                content="Legal analysis of liability limitations and risk allocation mechanisms.",
                doc_type="case_law",
                jurisdiction="UK",
                date="2023-02-15",
                source_url="test://sample2.pdf",
                similarity=0.88
            ),
            RAGContextItem(
                chunk_id=125,
                content="Contract termination procedures and notice requirements under common law.",
                doc_type="legal_document",
                jurisdiction="CA",
                date="2023-03-10",
                source_url="test://sample3.pdf",
                similarity=0.82
            )
        ]

    @pytest.mark.asyncio
    async def test_retrieve_contexts_success(self, sample_rag_contexts):
        """Test successful context retrieval"""
        # Mock the global RAG service to prevent initialization issues
        with patch('src.backend.services.rag._rag_service') as mock_rag_service:
            mock_rag_service.initialized = True
            
            # Create a mock service instance
            mock_service = AsyncMock()
            mock_service.retrieve_contexts.return_value = sample_rag_contexts
            
            with patch('src.backend.services.rag.get_rag_service', return_value=mock_service):
                result = await rag.retrieve_contexts("indemnification clause", top_k=5)
                
                assert len(result) > 0
                assert isinstance(result[0], RAGContextItem)
                assert result[0].similarity > 0.9
                mock_service.retrieve_contexts.assert_called_once_with("indemnification clause", 5)

    @pytest.mark.asyncio
    async def test_retrieve_contexts_embedding_failure(self):
        """Test context retrieval when embedding fails"""
        with patch('src.backend.services.rag.embed_texts_async') as mock_embed:
            mock_embed.return_value = []  # Empty embeddings
            
            mock_service = AsyncMock()
            mock_service.retrieve_contexts.return_value = []
            
            with patch('src.backend.services.rag.get_rag_service', return_value=mock_service):
                result = await rag.retrieve_contexts("test query")
                
                assert len(result) == 0

    @pytest.mark.asyncio
    async def test_summarize_200w_success(self):
        """Test successful text summarization"""
        mock_service = AsyncMock()
        mock_service.summarize_200w.return_value = "This is a legal contract with key terms including liability, termination, and payment provisions."
        
        with patch('src.backend.services.rag.get_rag_service', return_value=mock_service):
            result = await rag.summarize_200w("Long legal document text here...")
            
            assert len(result) > 0
            assert len(result.split()) <= 200
            mock_service.summarize_200w.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_200w_vertex_failure(self):
        """Test summarization fallback when Vertex AI fails"""
        mock_service = AsyncMock()
        mock_service.summarize_200w.return_value = "This legal document contains 150 words covering contractual terms and conditions..."
        
        with patch('src.backend.services.rag.get_rag_service', return_value=mock_service):
            result = await rag.summarize_200w("Short text")
            
            assert len(result) > 0
            mock_service.summarize_200w.assert_called_once()

    @pytest.mark.asyncio
    async def test_answer_with_vertex_success(self, sample_rag_contexts):
        """Test answer generation with context"""
        mock_service = AsyncMock()
        mock_service.answer_with_vertex.return_value = "Based on the legal precedents, indemnification clauses typically protect one party from claims."
        
        with patch('src.backend.services.rag.get_rag_service', return_value=mock_service):
            result = await rag.answer_with_vertex(
                "What is an indemnification clause?",
                sample_rag_contexts,
                "Contract analysis"
            )
            
            assert len(result) > 0
            assert "indemnification" in result.lower()
            mock_service.answer_with_vertex.assert_called_once_with(
                "What is an indemnification clause?",
                sample_rag_contexts,
                "Contract analysis"
            )

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test RAG service health check"""
        expected_health = {
            "status": "healthy",
            "initialized": True,
            "milvus_connected": True,
            "vertex_initialized": True
        }
        
        with patch('src.backend.services.rag.get_rag_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.initialized = True
            mock_service.get_service_stats.return_value = expected_health
            mock_get_service.return_value = mock_service
            
            result = await rag.health_check()
            
            assert "status" in result
            assert result["status"] in ["healthy", "degraded", "unhealthy"]


class TestMilvusClient:
    @pytest.mark.asyncio
    async def test_milvus_connection_success(self):
        """Test successful Milvus connection"""
        with patch('src.backend.services.rag.PYMILVUS_AVAILABLE', True):
            with patch('src.backend.services.rag.connections') as mock_conn, \
                 patch('src.backend.services.rag.utility') as mock_util, \
                 patch('src.backend.services.rag.Collection') as mock_collection:
                
                mock_util.has_collection.return_value = True
                mock_collection_instance = MagicMock()
                mock_collection.return_value = mock_collection_instance
                
                client = rag.MilvusClient()
                await client.initialize()
                
                assert client.connected is True
                mock_conn.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_milvus_search_success(self):
        """Test successful Milvus search"""
        client = rag.MilvusClient()
        client.connected = True
        client.collection = MagicMock()
        
        # Mock search results
        mock_hit = MagicMock()
        mock_hit.id = 123
        mock_hit.distance = 0.1
        mock_hit.entity = {
            "content": "Legal text",
            "doc_type": "contract", 
            "jurisdiction": "US"
        }
        
        mock_results = [[mock_hit]]
        
        # ✅ FIXED: Use AsyncMock for run_in_executor
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_event_loop = MagicMock()
            mock_loop.return_value = mock_event_loop
            # Use AsyncMock to properly handle await
            mock_event_loop.run_in_executor = AsyncMock(return_value=mock_results)
            
            query_vector = [0.1] * 768
            results = await client.search(query_vector, top_k=5)
            
            assert len(results) > 0
            assert results[0]["similarity"] > 0.8
            assert results[0]["content"] == "Legal text"
            assert results[0]["id"] == 123



class TestVertexAIClient:
    @pytest.mark.asyncio
    async def test_vertex_initialization_success(self):
        """Test successful Vertex AI initialization"""
        with patch('src.backend.services.rag.VERTEX_AI_AVAILABLE', True):
            with patch('src.backend.services.rag.aiplatform') as mock_aiplatform:
                client = rag.VertexAIClient()
                await client.initialize()
                
                assert client.initialized is True
                mock_aiplatform.init.assert_called_once()

    @pytest.mark.asyncio
    async def test_vertex_generation_fallback(self):
        """Test Vertex AI fallback response"""
        client = rag.VertexAIClient()
        client.initialized = False  # Force fallback
        
        result = await client.generate_text("Explain this contract")
        
        assert len(result) > 0
        assert "legal" in result.lower() or "contract" in result.lower()
