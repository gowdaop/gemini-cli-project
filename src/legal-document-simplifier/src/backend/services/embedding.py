import asyncio
import logging
from typing import List, Optional, Dict, Any
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
import time
from collections import OrderedDict

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    import numpy as np
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoTokenizer = AutoModel = torch = np = None
    TRANSFORMERS_AVAILABLE = False

# Configure logging - FIXED: Use __name__ instead of name
logger = logging.getLogger(__name__)

# Configuration
@dataclass
class EmbeddingConfig:
    model_name: str = "law-ai/InLegalBERT"  # ✅ Changed from nlpaueb/legal-bert-base-uncased
    embedding_dimension: int = 768
    max_sequence_length: int = 512
    batch_size: int = 8
    max_workers: int = 2
    cache_size: int = 1000
    cache_ttl: int = 3600  # 1 hour
    use_gpu: bool = True
    model_cache_dir: Optional[str] = None

class LRUCacheWithTTL:
    """LRU Cache with TTL (Time To Live) support"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        
        # Check TTL
        if time.time() - self.timestamps[key] > self.ttl:
            self._remove(key)
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def set(self, key: str, value: Any) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                oldest = next(iter(self.cache))
                self._remove(oldest)
            
            self.cache[key] = value
        
        self.timestamps[key] = time.time()
    
    def _remove(self, key: str) -> None:
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
    
    def clear(self) -> None:
        self.cache.clear()
        self.timestamps.clear()
    
    def size(self) -> int:
        return len(self.cache)

class EmbeddingService:
    """Optimized embedding service for legal documents"""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._model = None
        self._tokenizer = None
        self._device = None
        self._executor = None
        self._cache = LRUCacheWithTTL(
            max_size=self.config.cache_size,
            ttl=self.config.cache_ttl
        )
        self._model_loaded = False
        
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
    
    async def initialize(self) -> None:
        """Initialize the service asynchronously"""
        if not self._executor:
            self._executor = ThreadPoolExecutor(
                max_workers=self.config.max_workers,
                thread_name_prefix="embedding"
            )
        
        # Load model in background
        if TRANSFORMERS_AVAILABLE and not self._model_loaded:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self._load_model)
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        
        # Clear GPU memory
        if self._model and hasattr(self._model, 'cpu'):
            self._model.cpu()
            del self._model
            
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._cache.clear()
    
    @property
    def is_available(self) -> bool:
        """Check if the embedding service is available"""
        return TRANSFORMERS_AVAILABLE
    
    def _determine_device(self) -> str:
        """Determine the best device for model execution"""
        if not self.config.use_gpu or not torch or not torch.cuda.is_available():
            return "cpu"
        
        # Check GPU memory
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory < 2e9:  # Less than 2GB
                logger.warning("GPU memory insufficient, using CPU")
                return "cpu"
            return "cuda"
        except Exception:
            return "cpu"
    
    def _load_model(self) -> None:
        """Load the model and tokenizer with optimizations"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, using fallback")
            return
        
        try:
            logger.info(f"Loading model: {self.config.model_name}")
            
            # Set cache directory if specified
            cache_dir = self.config.model_cache_dir or os.getenv("TRANSFORMERS_CACHE")
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=cache_dir,
                use_fast=True  # Use fast tokenizer if available
            )
            
            # Load model with optimizations
            model_kwargs = {
                'cache_dir': cache_dir,
                'low_cpu_mem_usage': True
            }
            
            # Add torch_dtype only if torch is available
            if torch and torch.cuda.is_available():
                model_kwargs['torch_dtype'] = torch.float16
            
            self._model = AutoModel.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            
            # Set device
            self._device = self._determine_device()
            if self._device == "cuda" and torch:
                self._model = self._model.cuda()
                logger.info("Model loaded on GPU with half precision")
            else:
                logger.info("Model loaded on CPU")
            
            self._model.eval()
            self._model_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._model = None
            self._tokenizer = None
    
    def _get_cache_key(self, texts: List[str], prefix: str = "embed") -> str:
        """Generate efficient cache key"""
        # Use first 100 chars of each text to avoid huge keys
        text_sample = "|".join(text[:100] for text in texts[:10])  # Limit to first 10 texts
        text_hash = hashlib.blake2b(
            text_sample.encode(), 
            digest_size=16
        ).hexdigest()
        return f"{prefix}_{len(texts)}_{text_hash}"
    
    def _create_fallback_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create improved deterministic fallback embeddings"""
        embeddings = []
        for text in texts:
            # Create more sophisticated hash-based embedding
            text_bytes = text.encode('utf-8')
            
            # Use multiple hash functions for better distribution
            hashes = [
                hashlib.md5(text_bytes).digest(),
                hashlib.sha1(text_bytes).digest()[:16],
                hashlib.blake2b(text_bytes, digest_size=16).digest()
            ]
            
            # Combine hashes and normalize
            combined = b''.join(hashes)
            embedding = [(b / 255.0 - 0.5) * 2 for b in combined]  # Scale to [-1, 1]
            
            # Pad or truncate to required dimension
            if len(embedding) < self.config.embedding_dimension:
                # Use text statistics for padding
                text_stats = [
                    len(text) / 1000.0,  # Length feature
                    text.count(' ') / len(text) if text else 0,  # Space ratio
                    sum(c.isupper() for c in text) / len(text) if text else 0  # Upper case ratio
                ]
                padding_size = self.config.embedding_dimension - len(embedding) - len(text_stats)
                embedding.extend(text_stats)
                embedding.extend([0.0] * padding_size)
            else:
                embedding = embedding[:self.config.embedding_dimension]
                
            embeddings.append(embedding)
        
        return embeddings
    
    def _encode_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous batch encoding with memory management"""
        if not self._model or not self._tokenizer:
            return self._create_fallback_embeddings(texts)
        
        try:
            # Tokenize efficiently
            encoded = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_sequence_length,
                return_tensors="pt",
                add_special_tokens=True
            )
            
            # Move to device
            if self._device == "cuda" and torch:
                encoded = {k: v.cuda(non_blocking=True) for k, v in encoded.items()}
            
            # Generate embeddings with memory management
            with torch.no_grad():
                outputs = self._model(**encoded)
                
                # Use CLS token embedding or mean pooling
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    embeddings = outputs.pooler_output
                else:
                    # Mean pooling over sequence dimension, excluding padding
                    attention_mask = encoded.get('attention_mask')
                    if attention_mask is not None:
                        mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size())
                        sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
                        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                        embeddings = sum_embeddings / sum_mask
                    else:
                        embeddings = outputs.last_hidden_state.mean(dim=1)
                
                # Normalize embeddings
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                # Move to CPU and convert efficiently
                embeddings_cpu = embeddings.cpu()
                result = embeddings_cpu.numpy().tolist()
                
                # Clean up GPU memory
                if self._device == "cuda" and torch:
                    del outputs, embeddings, embeddings_cpu
                    torch.cuda.empty_cache()
                
                return result
                
        except Exception as e:
            logger.error(f"Encoding error: {e}")
            return self._create_fallback_embeddings(texts)
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Async embedding generation with caching"""
        if not texts:
            return []
        
        # Ensure service is initialized
        if not self._executor:
            await self.initialize()
        
        # Check cache
        cache_key = self._get_cache_key(texts)
        cached_result = self._cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for {len(texts)} texts")
            return cached_result
        
        # Process in batches
        all_embeddings = []
        
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            batch_embeddings = await loop.run_in_executor(
                self._executor,
                self._encode_batch_sync,
                batch
            )
            
            all_embeddings.extend(batch_embeddings)
        
        # Cache results
        self._cache.set(cache_key, all_embeddings)
        
        logger.info(f"Generated embeddings for {len(texts)} texts")
        return all_embeddings
    
    def embed_texts_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous embedding generation"""
        if not texts:
            return []
        
        if not self._model_loaded and TRANSFORMERS_AVAILABLE:
            self._load_model()
        
        cache_key = self._get_cache_key(texts)
        cached_result = self._cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        all_embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            batch_embeddings = self._encode_batch_sync(batch)
            all_embeddings.extend(batch_embeddings)
        
        self._cache.set(cache_key, all_embeddings)
        return all_embeddings
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            "model_name": self.config.model_name,
            "embedding_dimension": self.config.embedding_dimension,
            "max_sequence_length": self.config.max_sequence_length,
            "batch_size": self.config.batch_size,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "model_loaded": self._model_loaded,
            "device": self._device,
            "cache_size": self._cache.size(),
            "cache_max_size": self.config.cache_size,
            "gpu_available": torch.cuda.is_available() if torch else False
        }

# Global service instance with proper lifecycle management
_embedding_service = None

async def get_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
        await _embedding_service.initialize()
    return _embedding_service

# Async context manager for service lifecycle
@asynccontextmanager
async def embedding_service(config: Optional[EmbeddingConfig] = None):
    """Context manager for embedding service"""
    service = EmbeddingService(config)
    try:
        await service.initialize()
        yield service
    finally:
        await service.cleanup()

# Public API functions
async def embed_texts_async(texts: List[str]) -> List[List[float]]:
    """Async embedding generation - recommended for FastAPI"""
    service = await get_embedding_service()
    return await service.embed_texts(texts)

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Synchronous embedding generation - backward compatible"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service.embed_texts_sync(texts)

async def get_embedding_stats() -> Dict[str, Any]:
    """Get embedding service statistics"""
    service = await get_embedding_service()
    return service.get_stats()

def get_embedding_info() -> Dict[str, Any]:
    """Backward compatible stats function"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service.get_stats()

def health_check() -> Dict[str, Any]:
    """Health check for the embedding service"""
    try:
        test_result = embed_texts(["health check test"])
        return {
            "status": "healthy",
            "embedding_dimension": len(test_result[0]) if test_result else 0,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "timestamp": time.time()
        }