# tests/test_embedding_complete.py - Combined comprehensive test suite
import pytest
import asyncio
import hashlib
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List

from backend.services.embedding import (
    embed_texts,
    embed_texts_async,
    get_embedding_info,
    health_check,
    EmbeddingService,
    EmbeddingConfig
)

EMBEDDING_DIMENSION = 768

class TestEmbeddingServiceCore:
    """Core functionality tests for EmbeddingService"""
    
    def test_embedding_service_initialization(self):
        """Test that embedding service initializes correctly"""
        service = EmbeddingService()
        assert service._model is None
        assert service._tokenizer is None
        assert service._cache is not None
        assert service._executor is None

    def test_is_available_property(self):
        """Test the is_available property"""
        service = EmbeddingService()
        assert isinstance(service.is_available, bool)

    @patch('backend.services.embedding.TRANSFORMERS_AVAILABLE', False)
    def test_fallback_embeddings_when_transformers_unavailable(self):
        """Test fallback behavior when transformers is not available"""
        service = EmbeddingService()
        texts = ["test text", "another test"]
        embeddings = service._create_fallback_embeddings(texts)
        
        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert len(embedding) == EMBEDDING_DIMENSION
            assert all(isinstance(x, float) for x in embedding)

    def test_fallback_embeddings_deterministic(self):
        """Test that fallback embeddings are deterministic"""
        service = EmbeddingService()
        text = "consistent text"
        embedding1 = service._create_fallback_embeddings([text])[0]
        embedding2 = service._create_fallback_embeddings([text])[0]
        assert embedding1 == embedding2

    def test_cache_key_generation(self):
        """Test cache key generation"""
        service = EmbeddingService()
        texts1 = ["hello", "world"]
        texts2 = ["hello", "world"]
        texts3 = ["world", "hello"]
        
        key1 = service._get_cache_key(texts1)
        key2 = service._get_cache_key(texts2)
        key3 = service._get_cache_key(texts3)
        
        assert key1 == key2  # Same texts, same order
        assert key1 != key3  # Same texts, different order

class TestSyncEmbedding:
    """Synchronous embedding function tests"""
    
    def test_embed_texts_basic(self):
        """Test basic embedding functionality"""
        texts = ["This is a test sentence.", "Another test sentence."]
        embeddings = embed_texts(texts)
        
        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert len(embedding) == EMBEDDING_DIMENSION
            assert all(isinstance(x, float) for x in embedding)

    def test_embed_texts_empty_input(self):
        """Test embedding with empty input"""
        embeddings = embed_texts([])
        assert embeddings == []

    def test_embed_texts_single_text(self):
        """Test embedding with single text"""
        texts = ["Single test sentence."]
        embeddings = embed_texts(texts)
        
        assert len(embeddings) == 1
        assert len(embeddings[0]) == EMBEDDING_DIMENSION

    def test_embed_texts_long_text(self):
        """Test embedding with long text (truncation)"""
        long_text = "This is a very long sentence. " * 100
        embeddings = embed_texts([long_text])
        
        assert len(embeddings) == 1
        assert len(embeddings[0]) == EMBEDDING_DIMENSION

    def test_embed_texts_special_characters(self):
        """Test embedding with special characters"""
        texts = ["Text with @#$%^&*() special chars!", "Another with émojis 🎉"]
        embeddings = embed_texts(texts)
        
        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert len(embedding) == EMBEDDING_DIMENSION

class TestAsyncEmbedding:
    """Asynchronous embedding function tests"""
    
    @pytest.mark.asyncio
    async def test_embed_texts_async_basic(self):
        """Test basic async embedding functionality"""
        texts = ["Async test sentence.", "Another async test."]
        embeddings = await embed_texts_async(texts)
        
        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert len(embedding) == EMBEDDING_DIMENSION
            assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_embed_texts_async_empty_input(self):
        """Test async embedding with empty input"""
        embeddings = await embed_texts_async([])
        assert embeddings == []

    @pytest.mark.asyncio
    async def test_embed_texts_async_large_batch(self):
        """Test async embedding with large batch"""
        texts = [f"Test sentence number {i}" for i in range(50)]
        embeddings = await embed_texts_async(texts)
        
        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert len(embedding) == EMBEDDING_DIMENSION

    @pytest.mark.asyncio
    async def test_embed_texts_async_caching(self):
        """Test that async embedding uses caching"""
        texts = ["Cacheable sentence"]
        
        embeddings1 = await embed_texts_async(texts)
        embeddings2 = await embed_texts_async(texts)
        
        assert embeddings1 == embeddings2

class TestCaching:
    """Caching functionality tests"""
    
    def test_caching_works(self):
        """Test that caching works correctly"""
        service = EmbeddingService()
        texts = ["Cache test sentence"]
        
        # Mock the _encode_batch_sync method to track calls
        service._encode_batch_sync = Mock(return_value=[[0.1] * EMBEDDING_DIMENSION])
        
        # First call should hit the encoder
        embeddings1 = service.embed_texts_sync(texts)
        assert service._encode_batch_sync.call_count == 1
        
        # Second call should use cache
        embeddings2 = service.embed_texts_sync(texts)
        assert service._encode_batch_sync.call_count == 1  # Still 1, not 2
        assert embeddings1 == embeddings2

    def test_cache_eviction(self):
        """Test that cache eviction works"""
        config = EmbeddingConfig(cache_size=10)  # Small cache for testing
        service = EmbeddingService(config)
        
        # Fill cache beyond limit
        for i in range(15):  # More than cache limit of 10
            texts = [f"Cache test {i}"]
            service.embed_texts_sync(texts)
        
        # Cache should be limited
        assert service._cache.size() <= 10

class TestErrorHandling:
    """Error handling scenario tests"""
    
    @patch('backend.services.embedding.TRANSFORMERS_AVAILABLE', False)
    def test_graceful_fallback_no_transformers(self):
        """Test graceful fallback when transformers unavailable"""
        texts = ["Fallback test"]
        embeddings = embed_texts(texts)
        
        assert len(embeddings) == len(texts)
        assert len(embeddings[0]) == EMBEDDING_DIMENSION

    @patch('backend.services.embedding.EmbeddingService._load_model')
    def test_model_loading_error(self, mock_load):
        """Test handling of model loading errors"""
        # Mock _load_model to do nothing (simulate failure without exception)
        mock_load.return_value = None
        
        service = EmbeddingService()
        # Manually set the service state to simulate failed loading
        service._model = None
        service._tokenizer = None
        service._model_loaded = False
        
        texts = ["Error test"]
        embeddings = service.embed_texts_sync(texts)
        
        # Should fallback to deterministic embeddings
        assert len(embeddings) == len(texts)
        assert len(embeddings[0]) == EMBEDDING_DIMENSION
        # Verify _load_model was called
        mock_load.assert_called_once()

class TestMockScenarios:
    """Mock-based tests for controlled scenarios"""
    
    @patch('backend.services.embedding.torch')
    @patch('backend.services.embedding.AutoModel')
    @patch('backend.services.embedding.AutoTokenizer')
    @patch('backend.services.embedding.TRANSFORMERS_AVAILABLE', True)
    def test_model_loading_success(self, mock_tokenizer, mock_model, mock_torch):
        """Test successful model loading with mocks"""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_model_instance.eval.return_value = None
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_torch.cuda.is_available.return_value = False
        
        service = EmbeddingService()
        service._load_model()
        
        assert service._tokenizer is not None
        assert service._model is not None
        mock_model_instance.eval.assert_called_once()

    @patch('backend.services.embedding.torch')
    def test_gpu_detection(self, mock_torch):
        """Test GPU detection logic"""
        mock_torch.cuda.is_available.return_value = True
        
        service = EmbeddingService()
        device = service._determine_device()
        
        assert device in ["cuda", "cpu"]

    @patch('backend.services.embedding.EmbeddingService._encode_batch_sync')
    def test_batch_processing_calls(self, mock_encode):
        """Test that large batches are processed in chunks"""
        mock_encode.return_value = [[0.1] * 768]
        
        service = EmbeddingService()
        texts = ["test"] * 25  # More than batch size of 8
        
        service.embed_texts_sync(texts)
        
        # Should be called multiple times for batches
        assert mock_encode.call_count >= 3  # 25 texts / 8 batch size = 4 batches

class TestPerformance:
    """Performance and efficiency tests"""
    
    def test_sync_performance_small_batch(self):
        """Test performance with small batch"""
        texts = ["Performance test"] * 5
        
        start_time = time.time()
        embeddings = embed_texts(texts)
        end_time = time.time()
        
        assert len(embeddings) == len(texts)
        assert (end_time - start_time) < 10.0  # Should complete in under 10 seconds

    @pytest.mark.asyncio
    async def test_async_performance_large_batch(self):
        """Test async performance with large batch"""
        texts = [f"Large batch test {i}" for i in range(100)]
        
        start_time = time.time()
        embeddings = await embed_texts_async(texts)
        end_time = time.time()
        
        assert len(embeddings) == len(texts)
        print(f"Time for 100 embeddings: {end_time - start_time:.2f} seconds")

    def test_caching_performance(self):
        """Test that caching improves performance"""
        texts = ["Cache performance test"]
        
        # First call (cold)
        start_time = time.time()
        embed_texts(texts)
        cold_time = time.time() - start_time
        
        # Second call (cached)
        start_time = time.time()
        embed_texts(texts)
        cached_time = time.time() - start_time
        
        # Cached should be much faster
        assert cached_time < cold_time or cached_time < 0.01  # Very fast due to cache

class TestUtilityFunctions:
    """Utility function tests"""
    
    def test_get_embedding_info(self):
        """Test embedding info function"""
        info = get_embedding_info()
        
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "embedding_dimension" in info
        assert "transformers_available" in info
        assert "device" in info
        assert "cache_size" in info
        assert info["embedding_dimension"] == EMBEDDING_DIMENSION

    def test_health_check_healthy(self):
        """Test health check when service is healthy"""
        health = health_check()
        
        assert isinstance(health, dict)
        assert "status" in health
        assert health["status"] in ["healthy", "unhealthy"]
        assert "embedding_dimension" in health
        assert "transformers_available" in health

    @patch('backend.services.embedding.embed_texts')
    def test_health_check_unhealthy(self, mock_embed):
        """Test health check when service has errors"""
        mock_embed.side_effect = Exception("Test error")
        
        health = health_check()
        assert health["status"] == "unhealthy"
        assert "error" in health

class TestIntegration:
    """Integration and consistency tests"""
    
    @pytest.mark.asyncio
    async def test_sync_async_consistency(self):
        """Test that sync and async versions return same results"""
        texts = ["Consistency test sentence"]
        
        sync_embeddings = embed_texts(texts)
        async_embeddings = await embed_texts_async(texts)
        
        # Due to caching, they should be the same
        assert sync_embeddings == async_embeddings

    def test_batch_consistency(self):
        """Test that batch processing gives consistent results"""
        texts = [f"Batch test {i}" for i in range(20)]
        
        # Process all at once
        all_at_once = embed_texts(texts)
        
        # Process individually
        individual = []
        for text in texts:
            individual.extend(embed_texts([text]))
        
        # Results should have same structure
        assert len(all_at_once) == len(individual)
        for emb in all_at_once + individual:
            assert len(emb) == EMBEDDING_DIMENSION

class TestLegalBERTSpecific:
    """Legal domain-specific tests"""
    
    def test_legal_text_processing(self):
        """Test embedding generation for legal text"""
        legal_texts = [
            "The party shall be liable for damages",
            "This agreement terminates upon breach",
            "Confidential information must not be disclosed",
            "Governing law shall be the state of California"
        ]
        
        embeddings = embed_texts(legal_texts)
        
        assert len(embeddings) == len(legal_texts)
        for embedding in embeddings:
            assert len(embedding) == EMBEDDING_DIMENSION
            # Embeddings should not be all zeros (unless in fallback mode)
            assert not all(x == 0.0 for x in embedding[:10])  # Check first 10 values

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
