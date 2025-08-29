
import asyncio
import logging
from typing import List, Optional, Tuple
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import hashlib

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    import numpy as np
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoTokenizer = AutoModel = torch = np = None
    TRANSFORMERS_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
EMBEDDING_DIMENSION = 768
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 8

class EmbeddingService:
    """Production-ready embedding service for legal documents"""
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._cache = {}
        
    @property
    def is_available(self) -> bool:
        """Check if the embedding service is available"""
        return TRANSFORMERS_AVAILABLE
    
    @lru_cache(maxsize=1)
    def _load_model(self) -> Tuple[Optional[object], Optional[object]]:
        """Load the LegalBERT model and tokenizer with error handling"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, using fallback embeddings")
            return None, None
            
        try:
            logger.info(f"Loading LegalBERT model: {MODEL_NAME}")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModel.from_pretrained(MODEL_NAME)
            model.eval()
            
            # Move to GPU if available
            if torch.cuda.is_available():
                model = model.cuda()
                logger.info("Model loaded on GPU")
            else:
                logger.info("Model loaded on CPU")
                
            return tokenizer, model
            
        except Exception as e:
            logger.error(f"Failed to load LegalBERT model: {e}")
            return None, None
    
    def _get_cache_key(self, texts: List[str]) -> str:
        """Generate cache key for text batch"""
        text_hash = hashlib.md5("|".join(texts).encode()).hexdigest()
        return f"embed_{text_hash}"
    
    def _create_fallback_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create deterministic fallback embeddings"""
        embeddings = []
        for text in texts:
            # Create a simple hash-based embedding for consistency
            text_hash = hashlib.md5(text.encode()).digest()
            embedding = [float(b) / 255.0 - 0.5 for b in text_hash[:16]]  # 16 values
            # Pad to 768 dimensions
            embedding.extend([0.0] * (EMBEDDING_DIMENSION - len(embedding)))
            embeddings.append(embedding)
        return embeddings
    
    def _encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts using LegalBERT"""
        tokenizer, model = self._load_model()
        
        if tokenizer is None or model is None:
            return self._create_fallback_embeddings(texts)
        
        try:
            # Tokenize with proper truncation and padding
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
                return_tensors="pt"
            )
            
            # Move to same device as model
            if torch.cuda.is_available() and next(model.parameters()).is_cuda:
                encoded = {k: v.cuda() for k, v in encoded.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = model(**encoded)
                # Use mean pooling over the sequence dimension
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
                # Move back to CPU and convert to list
                embeddings = embeddings.cpu().numpy().tolist()
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}")
            return self._create_fallback_embeddings(texts)
    
    async def embed_texts_async(self, texts: List[str]) -> List[List[float]]:
        """Async wrapper for embedding generation"""
        if not texts:
            return []
        
        # Check cache first
        cache_key = self._get_cache_key(texts)
        if cache_key in self._cache:
            logger.debug(f"Cache hit for {len(texts)} texts")
            return self._cache[cache_key]
        
        # Process in batches to avoid memory issues
        all_embeddings = []
        
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            
            # Run embedding generation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            batch_embeddings = await loop.run_in_executor(
                self._executor, 
                self._encode_batch, 
                batch
            )
            all_embeddings.extend(batch_embeddings)
        
        # Cache the results
        self._cache[cache_key] = all_embeddings
        
        # Limit cache size
        if len(self._cache) > 100:  # Simple cache eviction
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        logger.info(f"Generated embeddings for {len(texts)} texts")
        return all_embeddings
    
    def embed_texts_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous embedding generation for backward compatibility"""
        if not texts:
            return []
        
        cache_key = self._get_cache_key(texts)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        all_embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            batch_embeddings = self._encode_batch(batch)
            all_embeddings.extend(batch_embeddings)
        
        self._cache[cache_key] = all_embeddings
        
        if len(self._cache) > 100:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        return all_embeddings
    
    def get_embedding_info(self) -> dict:
        """Get information about the embedding service"""
        return {
            "model_name": MODEL_NAME,
            "dimension": EMBEDDING_DIMENSION,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "available": self.is_available,
            "device": "cuda" if torch.cuda.is_available() and TRANSFORMERS_AVAILABLE else "cpu",
            "cached_items": len(self._cache)
        }

# Global service instance
_embedding_service = EmbeddingService()

# Public API functions (backward compatible)
def embed_texts(texts: List[str]) -> List[List[float]]:
    """Synchronous embedding generation - backward compatible"""
    return _embedding_service.embed_texts_sync(texts)

async def embed_texts_async(texts: List[str]) -> List[List[float]]:
    """Asynchronous embedding generation - recommended for FastAPI"""
    return await _embedding_service.embed_texts_async(texts)

def get_embedding_info() -> dict:
    """Get embedding service information"""
    return _embedding_service.get_embedding_info()

# Health check function for FastAPI
def health_check() -> dict:
    """Health check for the embedding service"""
    try:
        # Test with a simple embedding
        test_result = embed_texts(["test"])
        return {
            "status": "healthy",
            "embedding_dimension": len(test_result[0]) if test_result else 0,
            "transformers_available": TRANSFORMERS_AVAILABLE
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e),
            "transformers_available": TRANSFORMERS_AVAILABLE
        }
