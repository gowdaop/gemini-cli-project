# tests/test_embedding_performance.py
import pytest
import time
import asyncio
from services.embedding import embed_texts, embed_texts_async

class TestEmbeddingPerformance:
    """Performance tests for embedding service"""
    
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
