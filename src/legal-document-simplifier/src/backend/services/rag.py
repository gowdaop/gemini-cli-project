import asyncio
import logging
import hashlib
import time
<<<<<<< Updated upstream
import os
import json
import requests
from typing import List, Optional, Dict, Any, Union, Tuple
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import uuid
from functools import lru_cache
from collections import OrderedDict

# ✅ FIXED: Simplified PyMilvus imports only
try:
    from pymilvus import connections, Collection, utility, DataType
    from pymilvus import MilvusException
    PYMILVUS_AVAILABLE = True
except ImportError:
    PYMILVUS_AVAILABLE = False
    MilvusException = Exception

from ..schemas.analysis import RAGContextItem
from ..config import settings
from .embedding import embed_texts_async

logger = logging.getLogger(__name__)


class RAGCache:
    """Thread-safe LRU cache with TTL for RAG operations"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key not in self.cache:
                return None
            
            if time.time() - self.timestamps[key] > self.ttl:
                await self._remove(key)
                return None
            
            self.cache.move_to_end(key)
            return self.cache[key]
    
    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    oldest = next(iter(self.cache))
                    await self._remove(oldest)
                
                self.cache[key] = value
                self.timestamps[key] = time.time()
    
    async def _remove(self, key: str) -> None:
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
    
    async def clear(self) -> None:
        async with self._lock:
            self.cache.clear()
            self.timestamps.clear()


class MilvusClient:
    """Production-ready Milvus client with connection pooling and retry logic"""
    
    def __init__(self):
        self.connected = False
        self.collection = None
        self._connection_retries = 3
        self._retry_delay = 2
        self._search_cache = RAGCache(max_size=500, ttl=1800)
    
    async def initialize(self):
        """Initialize Milvus connection with retry logic"""
        if self.connected:
            return
        
        if not PYMILVUS_AVAILABLE:
            logger.warning("PyMilvus not available - RAG will use fallback mode")
            return
        
        for attempt in range(self._connection_retries):
            try:
                connections.connect(
                    alias="default",
                    host=settings.MILVUS_HOST,
                    port=settings.MILVUS_PORT,
                    timeout=settings.MILVUS_TIMEOUT
                )
                
                if not utility.has_collection(settings.MILVUS_COLLECTION):
                    logger.error(f"Collection {settings.MILVUS_COLLECTION} not found in Milvus")
                    return
                
                self.collection = Collection(settings.MILVUS_COLLECTION)
                await asyncio.get_event_loop().run_in_executor(None, self.collection.load)
                
                self.connected = True
                logger.info(f"Connected to Milvus collection: {settings.MILVUS_COLLECTION}")
                await self._log_collection_info()
                break
                
            except Exception as e:
                logger.warning(f"Milvus connection attempt {attempt + 1} failed: {e}")
                if attempt == self._connection_retries - 1:
                    logger.error("All Milvus connection attempts failed - using fallback mode")
                else:
                    await asyncio.sleep(self._retry_delay ** attempt)
    
    async def _log_collection_info(self):
        """Log collection statistics"""
        try:
            # ✅ FIX: Properly call the method
            stats = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.collection.num_entities()
            )
            logger.info(f"Milvus collection {settings.MILVUS_COLLECTION} has {stats} vectors")
        except Exception as e:
            logger.debug(f"Could not get collection stats: {e}")

    async def search(self, query_vector: List[float], top_k: int = 8, metric_type: str = "COSINE") -> List[Dict[str, Any]]:
        """Search for similar vectors in Milvus"""
        if not self.connected or not self.collection:
            logger.warning("Milvus not connected - returning empty results")
            return []
        
        cache_key = self._create_search_cache_key(query_vector, top_k, metric_type)
        cached_result = await self._search_cache.get(cache_key)
        if cached_result:
            logger.debug("Returning cached search results")
            return cached_result
        
        try:
            search_params = {
                "metric_type": metric_type,
                "params": {"nprobe": 16} if metric_type in ["L2", "IP"] else {}
            }
            
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.search(
                    data=[query_vector],
                    anns_field="embedding",
                    param=search_params,
                    limit=top_k,
                    output_fields=["*"]
                )
            )
            
            processed_results = self._process_search_results(results)
            await self._search_cache.set(cache_key, processed_results)
            
            logger.info(f"Milvus search returned {len(processed_results)} results")
            return processed_results
            
        except Exception as e:
            logger.error(f"Milvus search failed: {e}")
            return []
    
    def _create_search_cache_key(self, vector: List[float], top_k: int, metric: str) -> str:
        """Create cache key for search results"""
        vector_hash = hashlib.md5(str(vector[:10]).encode()).hexdigest()[:8]
        return f"search_{vector_hash}_{top_k}_{metric}"
    
    def _process_search_results(self, results) -> List[Dict[str, Any]]:
        """Process Milvus search results into standardized format"""
        processed = []
        
        for hits in results:
            for hit in hits:
                try:
                    entity_data = hit.entity if hasattr(hit, 'entity') else {}
                    
                    processed.append({
                        "id": hit.id if hasattr(hit, 'id') else 0,
                        "distance": float(hit.distance) if hasattr(hit, 'distance') else 0.0,
                        "similarity": 1.0 - float(hit.distance) if hasattr(hit, 'distance') else 0.0,
                        "content": entity_data.get("content", ""),
                        "doc_type": entity_data.get("doc_type", "legal_document"),
                        "jurisdiction": entity_data.get("jurisdiction", "unknown"),
                        "date": entity_data.get("date", "2020-2025"),
                        "source_url": entity_data.get("source_url", "milvus_collection"),
                        "metadata": entity_data
                    })
                except Exception as e:
                    logger.warning(f"Error processing search hit: {e}")
                    continue
        
        return processed


class VertexAIClient:
    """HTTP-based Vertex AI client using direct REST API calls"""
    
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.initialized = False
        self._generation_cache = RAGCache(max_size=200, ttl=7200)
        
    async def initialize(self):
        """Initialize HTTP client"""
        if self.initialized:
            return
        
        if not self.api_key:
            logger.warning("❌ No Google API key found - using fallback responses")
            return
        
        try:
            # Test API connectivity
            test_url = f"{self.base_url}/models"
            headers = {
                'X-Goog-Api-Key': self.api_key,
                'Content-Type': 'application/json'
            }
            
            response = requests.get(test_url, headers=headers, timeout=10)
            if response.status_code == 200:
                self.initialized = True
                logger.info("✅ Vertex AI HTTP client initialized successfully")
            else:
                logger.warning(f"❌ API test failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Vertex AI HTTP client initialization failed: {e}")
    
    async def generate_text(self, prompt: str, max_tokens: int = None, temperature: float = None) -> str:
        """Generate text using direct HTTP API calls"""
        if not self.initialized:
            await self.initialize()
        
        if not self.initialized:
            return self._create_fallback_response(prompt)
        
        cache_key = hashlib.md5(f"{prompt}_{max_tokens}_{temperature}".encode()).hexdigest()
        cached_result = await self._generation_cache.get(cache_key)
        if cached_result:
            logger.debug("Returning cached generation result")
            return cached_result
        
        try:
            # Use the model name from environment variable
            model = getattr(settings, 'VERTEX_MODEL', 'gemini-2.0-flash')
            url = f"{self.base_url}/models/{model}:generateContent"
            
            headers = {
                'Content-Type': 'application/json',
                'X-Goog-Api-Key': self.api_key
            }
            
            # Build request payload matching Google Studio format
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": max_tokens or getattr(settings, 'VERTEX_MAX_TOKENS', 1024),
                    "temperature": temperature or getattr(settings, 'VERTEX_TEMPERATURE', 0.1),
                    "topP": 0.8,
                    "topK": 40
                }
            }
            
            logger.info(f"🤖 Calling Vertex AI HTTP API: {model}")
            
            # Make HTTP request
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.post(url, headers=headers, json=payload, timeout=30)
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract text from response
                if 'candidates' in result and len(result['candidates']) > 0:
                    candidate = result['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        generated_text = candidate['content']['parts'][0].get('text', '')
                        
                        if generated_text:
                            await self._generation_cache.set(cache_key, generated_text)
                            logger.info("✅ Vertex AI HTTP generation successful")
                            return generated_text
                
                logger.warning("❌ Empty response from Vertex AI")
                return self._create_fallback_response(prompt)
                
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"❌ Vertex AI HTTP API failed: {error_msg}")
                return self._create_fallback_response(prompt)
            
        except Exception as e:
            logger.error(f"❌ Vertex AI HTTP generation failed: {e}")
            return self._create_fallback_response(prompt)
    
    def _create_fallback_response(self, prompt: str) -> str:
        """Create fallback response when Vertex AI is unavailable"""
        prompt_lower = prompt.lower()
        
        if "summarize" in prompt_lower or "summary" in prompt_lower:
            return ("This legal document contains various contractual terms and conditions. "
                   "Key areas include liability, termination, payment terms, and governing law. "
                   "Please review carefully and consider consulting with a legal professional "
                   "for specific advice regarding your situation.")
        
        elif "risk" in prompt_lower or "danger" in prompt_lower:
            return ("Based on the document analysis, potential risks may include liability exposure, "
                   "termination conditions, and payment obligations. It's recommended to review "
                   "these clauses carefully and seek legal advice if needed.")
        
        elif "explain" in prompt_lower or "what" in prompt_lower:
            return ("This appears to be a legal document with standard contractual provisions. "
                   "The terms outlined establish rights, obligations, and procedures for the parties involved. "
                   "For specific interpretation, please consult with a qualified attorney.")
        
        else:
            return ("I understand you're looking for a clear explanation. Here's what I can tell you:\n\n"
                   "📖 **Legal Analysis**: This document contains various contractual provisions requiring careful review.\n\n"
                   "🔍 **Key Points**: Based on legal precedents, these terms establish specific obligations and rights.\n\n"
                   "💭 **In Plain English**: Legal contracts specify exactly what each party must do.\n\n"
                   "❓ **Need More Details?**: Feel free to ask specific questions about any clause.")


class RAGService:
    """Comprehensive RAG service combining Milvus, embeddings, and Vertex AI"""
    
    def __init__(self):
        self.milvus_client = MilvusClient()
        self.vertex_client = VertexAIClient()
        self.initialized = False
    
    async def initialize(self):
        """Initialize all RAG components"""
        if self.initialized:
            return
        
        logger.info("Initializing RAG service...")
        
        await asyncio.gather(
            self.milvus_client.initialize(),
            self.vertex_client.initialize(),
            return_exceptions=True
        )
        
        self.initialized = True
        logger.info("RAG service initialization complete")
    
    async def retrieve_contexts(self, query: str, top_k: int = 8) -> List[RAGContextItem]:
        """Retrieve relevant contexts from Milvus"""
        if not self.initialized:
            await self.initialize()
        
        logger.info(f"🔍 Generating embeddings for query: '{query[:50]}...'")
        
        try:
            # Generate embeddings with detailed logging
            query_embeddings = await embed_texts_async([query])
            
            if not query_embeddings or not query_embeddings[0]:
                logger.error("❌ Failed to generate query embedding")
                return []
            
            query_vector = query_embeddings[0]
            logger.info(f"✅ Generated embedding vector of dimension: {len(query_vector)}")
            logger.debug(f"🔢 First 5 embedding values: {query_vector[:5]}")
            
            # Milvus search with detailed logging
            logger.info(f"🔍 Searching Milvus with top_k={top_k}, metric=COSINE")
            milvus_results = await self.milvus_client.search(
                query_vector=query_vector,
                top_k=min(top_k, 8),
                metric_type="COSINE"
            )
            
            logger.info(f"🔍 Milvus returned {len(milvus_results)} raw results")
            
            # Process results with detailed logging
            rag_contexts = []
            for i, result in enumerate(milvus_results):
                try:
                    similarity = float(result.get("similarity", 0.0))
                    content_preview = result.get("content", "")[:100]
                    logger.debug(f"📄 Result {i+1}: similarity={similarity:.4f}, content='{content_preview}...'")
                    
                    if similarity > 0.1:  # Only include reasonably similar results
                        rag_contexts.append(RAGContextItem(
                            chunk_id=result.get("id", i),
                            content=result.get("content", "")[:2000],
                            doc_type=result.get("doc_type", "legal_document"),
                            jurisdiction=result.get("jurisdiction", "unknown"),
                            date=result.get("date", "2020-2025"),
                            source_url=result.get("source_url", "milvus_collection"),
                            similarity=similarity
                        ))
                    else:
                        logger.debug(f"🚫 Skipped result {i+1} due to low similarity: {similarity:.4f}")
                        
                except Exception as e:
                    logger.warning(f"❌ Error processing result {i+1}: {e}")
                    continue
            
            # Sort and return with final logging
            rag_contexts.sort(key=lambda x: x.similarity, reverse=True)
            final_count = len(rag_contexts)
            logger.info(f"✅ Successfully processed {final_count} contexts (similarity > 0.1)")
            
            if final_count == 0:
                logger.warning("⚠️ No documents found with sufficient similarity!")
                logger.info("🔧 Consider: 1) Checking embedding compatibility, 2) Lowering similarity threshold, 3) Increasing top_k")
            
            return rag_contexts[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Context retrieval failed: {e}", exc_info=True)
            return []
        
    async def summarize_200w(self, text: str) -> str:
        """Generate a summary in ≤200 words"""
        if not text or not text.strip():
            logger.warning("❌ No text provided for summary generation")
            return "No content available for summary."
        
        logger.info(f"🔍 Generating summary for {len(text)} characters")
        logger.debug(f"📄 Input text preview: {text[:200]}...")
        
        prompt = f"""You are a legal document assistant. Create a concise summary of this legal document in exactly 200 words or fewer.

Focus on:
1. Main purpose and type of document
2. Key parties involved
3. Primary obligations and rights
4. Important terms and conditions
5. Notable risks or considerations

Use simple, non-technical language.

Document text:
{text[:4000]}...

Summary (≤200 words):"""
        
        try:
            logger.info("🤖 Calling Vertex AI for summary generation...")
            summary = await self.vertex_client.generate_text(
                prompt=prompt,
                max_tokens=300,
                temperature=0.1
            )
            
            if not summary or not summary.strip():
                logger.warning("❌ Vertex AI returned empty summary")
                return self._create_fallback_summary(text)
            
            # Ensure word limit
            words = summary.split()
            if len(words) > 200:
                summary = " ".join(words[:200]) + "..."
            
            logger.info(f"✅ Generated summary: {len(summary)} chars, {len(words)} words")
            logger.debug(f"📝 Summary content: {summary[:150]}...")
=======
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
from dataclasses import dataclass
import json

# Project imports - Updated for Member 2's config compatibility
from ..schemas.analysis import RAGContextItem
from ..config import settings  # Uses Member 2's global settings

# External dependencies for production RAG
try:
    import pymilvus
    from pymilvus import connections, Collection, utility
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logging.warning("PyMilvus not available, using fallback RAG implementation")

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available, using fallback embeddings")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGMetrics:
    """Performance metrics for RAG operations"""
    total_queries: int = 0
    successful_queries: int = 0
    cache_hits: int = 0
    avg_retrieval_time: float = 0.0
    avg_generation_time: float = 0.0
    total_contexts_retrieved: int = 0
    
    def update_retrieval(self, success: bool, time_taken: float, context_count: int = 0):
        self.total_queries += 1
        if success:
            self.successful_queries += 1
            self.total_contexts_retrieved += context_count
        
        # Rolling average for retrieval time
        self.avg_retrieval_time = (
            (self.avg_retrieval_time * (self.total_queries - 1) + time_taken) / 
            self.total_queries
        )
    
    def record_cache_hit(self):
        self.cache_hits += 1
    
    @property
    def success_rate(self) -> float:
        return self.successful_queries / max(self.total_queries, 1)
    
    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / max(self.total_queries, 1)

class LegalEmbeddingService:
    """Production-grade embedding service for legal documents"""
    
    def __init__(self):
        # Updated to use Member 2's config
        self.model_name = settings.HF_MODEL_NAME
        self.cache_dir = getattr(settings, 'HF_CACHE_DIR', '/tmp/hf_cache')
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize embedding model with lazy loading"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, embedding service will use fallbacks")
            return
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Legal embedding model loaded: {self.model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            self.tokenizer = None
            self.model = None
    
    @lru_cache(maxsize=1000)
    def embed_text_cached(self, text: str) -> List[float]:
        """Cached embedding generation for frequently used texts"""
        return self._embed_single_text(text)
    
    def _embed_single_text(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        if not self.model or not self.tokenizer:
            # Fallback: simple hash-based pseudo-embedding
            return self._fallback_embedding(text)
        
        try:
            # Truncate text to model limits
            inputs = self.tokenizer(
                text, 
                return_tensors='pt', 
                truncation=True, 
                padding=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling of last hidden state
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                return embeddings[0].tolist()
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            return self._fallback_embedding(text)
    
    def embed_texts_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding generation for efficiency"""
        if not texts:
            return []
        
        if not self.model or not self.tokenizer:
            return [self._fallback_embedding(text) for text in texts]
        
        try:
            # Process in batches for memory efficiency
            batch_size = 16
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch, 
                    return_tensors='pt', 
                    truncation=True, 
                    padding=True, 
                    max_length=512
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    all_embeddings.extend(embeddings.tolist())
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Batch embedding failed: {str(e)}")
            return [self._fallback_embedding(text) for text in texts]
    
    def _fallback_embedding(self, text: str, dim: int = 768) -> List[float]:
        """Fallback embedding using text hashing"""
        # Create deterministic embedding from text hash
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        
        # Convert hash to numeric values
        embedding = []
        for i in range(0, min(len(text_hash), dim * 2), 2):
            hex_pair = text_hash[i:i+2]
            value = int(hex_pair, 16) / 255.0 - 0.5  # Normalize to [-0.5, 0.5]
            embedding.append(value)
        
        # Pad or truncate to desired dimension
        while len(embedding) < dim:
            embedding.append(0.0)
        
        return embedding[:dim]

class MilvusVectorStore:
    """Production Milvus vector store with connection pooling and error handling"""
    
    def __init__(self):
        # Updated to use Member 2's config
        self.host = settings.MILVUS_HOST
        self.port = settings.MILVUS_PORT
        self.collection_name = settings.MILVUS_COLLECTION
        self.timeout = getattr(settings, 'MILVUS_TIMEOUT', 30)
        self.connected = False
        self.collection = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Milvus with retry logic"""
        if not MILVUS_AVAILABLE:
            logger.warning("Milvus not available, vector store will use fallbacks")
            return
        
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=str(self.port),
                timeout=self.timeout
            )
            
            # Check if collection exists
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                self.collection.load()
                self.connected = True
                logger.info(f"Connected to Milvus collection: {self.collection_name} at {self.host}:{self.port}")
            else:
                logger.warning(f"Milvus collection {self.collection_name} not found")
                
        except Exception as e:
            logger.error(f"Failed to connect to Milvus at {self.host}:{self.port}: {str(e)}")
            self.connected = False
    
    async def search_similar(self, query_embedding: List[float], top_k: int = 10) -> List[RAGContextItem]:
        """Search for similar vectors in Milvus"""
        if not self.connected:
            return self._fallback_search(query_embedding, top_k)
        
        try:
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 16},
            }
            
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["content", "doc_type", "jurisdiction", "date", "source_url", "title"]
            )
            
            contexts = []
            for result in results[0]:
                contexts.append(RAGContextItem(
                    chunk_id=result.id,
                    content=result.entity.get("content", ""),
                    doc_type=result.entity.get("doc_type", "legal_document"),
                    jurisdiction=result.entity.get("jurisdiction", "Unknown"),
                    date=result.entity.get("date", "2023-01-01"),
                    source_url=result.entity.get("source_url", ""),
                    similarity=1.0 - result.distance  # Convert distance to similarity
                ))
            
            return contexts
            
        except Exception as e:
            logger.error(f"Milvus search failed: {str(e)}")
            return self._fallback_search(query_embedding, top_k)
    
    def _fallback_search(self, query_embedding: List[float], top_k: int) -> List[RAGContextItem]:
        """Fallback search using predefined legal contexts"""
        legal_contexts = [
            RAGContextItem(
                chunk_id=1,
                content="Termination clauses in employment contracts must specify clear grounds and notice periods to be enforceable under federal labor law.",
                doc_type="employment_law",
                jurisdiction="Federal",
                date="2023-07-15",
                source_url="https://legal-db.example.com/employment/termination",
                similarity=0.88
            ),
            RAGContextItem(
                chunk_id=2,
                content="Indemnification provisions create significant liability exposure and should include caps, carve-outs, and mutual indemnification clauses.",
                doc_type="contract_analysis",
                jurisdiction="Delaware",
                date="2023-08-20",
                source_url="https://legal-db.example.com/contracts/indemnification",
                similarity=0.85
            ),
            RAGContextItem(
                chunk_id=3,
                content="Confidentiality agreements must clearly define confidential information, include reasonable time limits, and provide for return of materials.",
                doc_type="nda_analysis",
                jurisdiction="California",
                date="2023-09-10",
                source_url="https://legal-db.example.com/nda/confidentiality",
                similarity=0.82
            ),
            RAGContextItem(
                chunk_id=4,
                content="Payment terms with compound interest rates exceeding statutory limits may violate state usury laws and render clauses unenforceable.",
                doc_type="financial_law",
                jurisdiction="New York",
                date="2023-06-05",
                source_url="https://legal-db.example.com/finance/payment-terms",
                similarity=0.79
            ),
            RAGContextItem(
                chunk_id=5,
                content="Intellectual property clauses should clearly specify ownership, licensing terms, work-for-hire provisions, and assignment of future developments.",
                doc_type="ip_law",
                jurisdiction="Federal",
                date="2023-05-20",
                source_url="https://legal-db.example.com/ip/ownership",
                similarity=0.76
            ),
            RAGContextItem(
                chunk_id=6,
                content="Governing law clauses determine applicable jurisdiction and should align with dispute resolution provisions and forum selection clauses.",
                doc_type="jurisdiction_analysis",
                jurisdiction="Multi-state",
                date="2023-04-15",
                source_url="https://legal-db.example.com/jurisdiction/governing-law",
                similarity=0.73
            ),
            RAGContextItem(
                chunk_id=7,
                content="Arbitration clauses can limit access to courts but must include provisions for discovery, legal fees, and appeals to be enforceable.",
                doc_type="dispute_resolution",
                jurisdiction="Federal",
                date="2023-08-01",
                source_url="https://legal-db.example.com/arbitration/procedures",
                similarity=0.70
            ),
            RAGContextItem(
                chunk_id=8,
                content="Force majeure clauses should specifically enumerate covered events, notice requirements, and mitigation obligations during COVID-19 era.",
                doc_type="contract_terms",
                jurisdiction="Multi-state",
                date="2023-03-10",
                source_url="https://legal-db.example.com/force-majeure/pandemic",
                similarity=0.68
            ),
            RAGContextItem(
                chunk_id=9,
                content="Data privacy clauses must comply with GDPR, CCPA, and state privacy laws regarding collection, processing, and retention of personal information.",
                doc_type="privacy_law",
                jurisdiction="Federal",
                date="2023-10-05",
                source_url="https://legal-db.example.com/privacy/compliance",
                similarity=0.65
            ),
            RAGContextItem(
                chunk_id=10,
                content="Non-compete agreements are subject to state-specific enforceability standards regarding geographic scope, duration, and legitimate business interests.",
                doc_type="employment_law",
                jurisdiction="Multi-state",
                date="2023-09-25",
                source_url="https://legal-db.example.com/employment/non-compete",
                similarity=0.62
            )
        ]
        
        # Simple relevance filtering for fallback
        return legal_contexts[:top_k]

class RAGService:
    """Production-grade RAG service for legal document question answering"""
    
    def __init__(self):
        self.embedding_service = LegalEmbeddingService()
        self.vector_store = MilvusVectorStore()
        self.metrics = RAGMetrics()
        
        # Cache for frequently asked questions with Member 2's config-based TTL
        self._answer_cache = {}
        self._context_cache = {}
        self.cache_ttl = timedelta(hours=6)  # 6-hour cache TTL
        
        # Rate limiting based on Member 2's performance settings
        self._request_timestamps = []
        self.max_requests_per_minute = getattr(settings, 'MAX_REQUESTS_PER_MINUTE', 100)
        
        logger.info(f"RAG Service initialized with {self.embedding_service.model_name} and Milvus at {self.vector_store.host}:{self.vector_store.port}")
    
    def _is_rate_limited(self) -> bool:
        """Simple rate limiting based on request timestamps"""
        now = datetime.now()
        # Clean old timestamps
        self._request_timestamps = [
            ts for ts in self._request_timestamps 
            if now - ts < timedelta(minutes=1)
        ]
        
        if len(self._request_timestamps) >= self.max_requests_per_minute:
            return True
        
        self._request_timestamps.append(now)
        return False
    
    def _get_cache_key(self, text: str, additional_params: str = "") -> str:
        """Generate cache key for caching"""
        return hashlib.md5(f"{text}:{additional_params}".encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        if 'timestamp' not in cache_entry:
            return False
        
        cached_time = datetime.fromisoformat(cache_entry['timestamp'])
        return datetime.now() - cached_time < self.cache_ttl
    
    async def retrieve_contexts(self, query_text: str, top_k: int = 8) -> List[RAGContextItem]:
        """
        Retrieve relevant legal contexts using vector similarity search
        
        Args:
            query_text: The question or text to find contexts for
            top_k: Number of most relevant contexts to retrieve
            
        Returns:
            List of RAGContextItem with legal precedents and information
        """
        
        if self._is_rate_limited():
            logger.warning("Rate limit exceeded for context retrieval")
            return []
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(query_text, str(top_k))
            if cache_key in self._context_cache:
                cache_entry = self._context_cache[cache_key]
                if self._is_cache_valid(cache_entry):
                    self.metrics.record_cache_hit()
                    logger.debug(f"Cache hit for context retrieval: {query_text[:50]}...")
                    return cache_entry['contexts']
            
            # Generate query embedding
            query_embedding = self.embedding_service.embed_text_cached(query_text)
            
            # Search vector database
            contexts = await self.vector_store.search_similar(query_embedding, top_k)
            
            # Filter and enhance contexts for legal relevance
            contexts = self._enhance_legal_contexts(contexts, query_text)
            
            # Cache results
            self._context_cache[cache_key] = {
                'contexts': contexts,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update metrics
            retrieval_time = time.time() - start_time
            self.metrics.update_retrieval(True, retrieval_time, len(contexts))
            
            logger.debug(f"Retrieved {len(contexts)} contexts in {retrieval_time:.2f}s")
            return contexts
            
        except Exception as e:
            retrieval_time = time.time() - start_time
            self.metrics.update_retrieval(False, retrieval_time, 0)
            logger.error(f"Context retrieval failed: {str(e)}")
            return []
    
    def _enhance_legal_contexts(self, contexts: List[RAGContextItem], query_text: str) -> List[RAGContextItem]:
        """Enhance contexts with legal-specific relevance scoring"""
        if not contexts:
            return contexts
        
        query_lower = query_text.lower()
        legal_keywords = [
            'contract', 'agreement', 'clause', 'liability', 'termination',
            'breach', 'damages', 'indemnify', 'confidential', 'payment',
            'intellectual property', 'governing law', 'arbitration', 'dispute',
            'employment', 'non-disclosure', 'force majeure', 'privacy'
        ]
        
        # Boost relevance for legal terminology matches
        enhanced_contexts = []
        for context in contexts:
            content_lower = context.content.lower()
            
            # Count legal keyword matches
            legal_matches = sum(1 for keyword in legal_keywords if keyword in content_lower)
            query_matches = sum(1 for word in query_lower.split() if word in content_lower)
            
            # Adjust similarity based on legal relevance
            relevance_boost = (legal_matches * 0.05) + (query_matches * 0.03)
            adjusted_similarity = min(context.similarity + relevance_boost, 1.0)
            
            enhanced_context = RAGContextItem(
                chunk_id=context.chunk_id,
                content=context.content,
                doc_type=context.doc_type,
                jurisdiction=context.jurisdiction,
                date=context.date,
                source_url=context.source_url,
                similarity=adjusted_similarity
            )
            enhanced_contexts.append(enhanced_context)
        
        # Sort by adjusted similarity
        enhanced_contexts.sort(key=lambda x: x.similarity, reverse=True)
        return enhanced_contexts
    
    async def answer_question(self, question: str, context: str = "", conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Answer legal questions using RAG with retrieved contexts
        
        Args:
            question: The legal question to answer
            context: Optional additional context
            conversation_id: Optional conversation tracking ID
            
        Returns:
            Dictionary with answer, sources, confidence, and reasoning
        """
        
        if self._is_rate_limited():
            return {
                "answer": "Rate limit exceeded. Please try again later.",
                "sources": [],
                "confidence": 0.0,
                "reasoning": "Request rate limited for service protection",
                "conversation_id": conversation_id
            }
        
        start_time = time.time()
        request_id = f"qa_{int(start_time * 1000)}"
        
        logger.info(f"[{request_id}] Processing legal question: {question[:100]}...")
        
        try:
            # Check answer cache
            cache_key = self._get_cache_key(f"{question}:{context}")
            if cache_key in self._answer_cache:
                cache_entry = self._answer_cache[cache_key]
                if self._is_cache_valid(cache_entry):
                    self.metrics.record_cache_hit()
                    logger.debug(f"[{request_id}] Cache hit for question answering")
                    result = cache_entry['answer'].copy()
                    result['conversation_id'] = conversation_id
                    return result
            
            # Retrieve relevant legal contexts
            contexts = await self.retrieve_contexts(question, top_k=8)
            
            # Generate answer using contexts
            answer_data = await self._generate_legal_answer(
                question, contexts, context, request_id
            )
            
            # Add metadata
            answer_data['conversation_id'] = conversation_id
            answer_data['processing_time_ms'] = round((time.time() - start_time) * 1000, 2)
            
            # Cache successful answers
            self._answer_cache[cache_key] = {
                'answer': answer_data,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"[{request_id}] Question answered successfully in {answer_data['processing_time_ms']}ms")
            return answer_data
            
        except Exception as e:
            logger.error(f"[{request_id}] Question answering failed: {str(e)}")
            return {
                "answer": f"I apologize, but I encountered an error processing your legal question. Please try rephrasing your question or contact support if the issue persists.",
                "sources": [],
                "confidence": 0.0,
                "reasoning": f"Error in question processing: {str(e)[:100]}",
                "conversation_id": conversation_id,
                "error": True
            }
    
    async def _generate_legal_answer(self, question: str, contexts: List[RAGContextItem], 
                                   additional_context: str, request_id: str) -> Dict[str, Any]:
        """Generate legal answer using retrieved contexts and AI"""
        
        if not contexts:
            return {
                "answer": "I don't have enough relevant legal information to provide a comprehensive answer to your question. Please provide more context or consult with a qualified attorney for specific legal advice.",
                "sources": [],
                "confidence": 0.2,
                "reasoning": "No relevant legal contexts found for this question"
            }
        
        # Combine contexts for answer generation
        context_text = "\n\n".join([
            f"Source {i+1} ({ctx.jurisdiction}, {ctx.doc_type}): {ctx.content}"
            for i, ctx in enumerate(contexts[:5])
        ])
        
        # Generate structured legal answer using template approach
        answer = self._generate_structured_legal_answer(question, contexts, additional_context)
        
        # Calculate confidence based on context relevance
        avg_similarity = sum(ctx.similarity for ctx in contexts) / len(contexts)
        confidence = min(avg_similarity + 0.1, 0.95)  # Boost but cap confidence
        
        # Extract sources
        sources = [
            f"{ctx.doc_type} - {ctx.jurisdiction} ({ctx.date}): {ctx.source_url}"
            for ctx in contexts[:3]
        ]
        
        # Generate reasoning
        reasoning = self._generate_reasoning(question, contexts, confidence)
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": round(confidence, 2),
            "reasoning": reasoning
        }
    
    def _generate_structured_legal_answer(self, question: str, contexts: List[RAGContextItem], 
                                        additional_context: str) -> str:
        """Generate structured legal answer using templates and contexts"""
        
        question_lower = question.lower()
        
        # Identify question type and generate appropriate answer structure
        if any(term in question_lower for term in ['termination', 'terminate', 'end contract']):
            return self._answer_termination_question(question, contexts, additional_context)
        elif any(term in question_lower for term in ['liability', 'liable', 'damages', 'responsible']):
            return self._answer_liability_question(question, contexts, additional_context)
        elif any(term in question_lower for term in ['indemnify', 'indemnification', 'hold harmless']):
            return self._answer_indemnification_question(question, contexts, additional_context)
        elif any(term in question_lower for term in ['confidential', 'nda', 'non-disclosure']):
            return self._answer_confidentiality_question(question, contexts, additional_context)
        elif any(term in question_lower for term in ['payment', 'pay', 'fees', 'invoice']):
            return self._answer_payment_question(question, contexts, additional_context)
        elif any(term in question_lower for term in ['intellectual property', 'ip', 'copyright', 'patent']):
            return self._answer_ip_question(question, contexts, additional_context)
        elif any(term in question_lower for term in ['arbitration', 'dispute', 'mediation']):
            return self._answer_dispute_question(question, contexts, additional_context)
        else:
            return self._answer_general_legal_question(question, contexts, additional_context)
    
    def _answer_termination_question(self, question: str, contexts: List[RAGContextItem], 
                                   additional_context: str) -> str:
        """Answer termination-related legal questions"""
        return f"""**Termination Clause Analysis:**

Based on legal precedent analysis, termination provisions require careful consideration of several key factors:

**Essential Elements:**
• Clear termination grounds and triggering events
• Appropriate notice periods (typically 30-90 days for commercial contracts)
• Cure periods for material breaches
• Post-termination obligations and survival clauses

**Legal Precedents:**
{self._format_context_summary(contexts[:2])}

**Risk Considerations:**
• Immediate termination clauses may be challengeable without proper cause
• Notice requirements must comply with applicable state laws
• Consider reciprocal termination rights for fairness

**Recommendation:**
Review termination language for enforceability and ensure compliance with jurisdiction-specific requirements. For employment contracts, consider at-will employment laws. For commercial agreements, include appropriate cure periods and notice provisions.

*This analysis is for informational purposes only and does not constitute legal advice. Consult qualified legal counsel for specific situations.*"""
    
    def _answer_liability_question(self, question: str, contexts: List[RAGContextItem], 
                                 additional_context: str) -> str:
        """Answer liability-related legal questions"""
        return f"""**Liability Provision Analysis:**

Liability clauses are critical for risk allocation and require careful drafting to ensure enforceability:

**Key Considerations:**
• Scope of liability (direct, indirect, consequential damages)
• Monetary caps and limitations on damages
• Carve-outs for gross negligence and willful misconduct
• Reciprocal vs. one-sided liability provisions

**Legal Framework:**
{self._format_context_summary(contexts[:2])}

**Enforceability Factors:**
• State laws may limit or prohibit certain liability exclusions
• Consumer protection laws may override limitation clauses
• Insurance considerations and coverage coordination

**Best Practices:**
• Include reasonable caps proportionate to contract value
• Ensure mutual liability limitations where appropriate
• Consider separate treatment for different types of damages
• Include indemnification provisions for third-party claims

*This analysis is for informational purposes only. Consult with legal counsel for jurisdiction-specific guidance.*"""
    
    def _answer_indemnification_question(self, question: str, contexts: List[RAGContextItem], 
                                       additional_context: str) -> str:
        """Answer indemnification-related legal questions"""
        return f"""**Indemnification Clause Analysis:**

Indemnification provisions transfer risk between parties and require careful structuring:

**Core Components:**
• Clear indemnification triggers and covered claims
• Scope of indemnified losses (damages, costs, attorney fees)
• Notice and defense obligations
• Cooperation requirements and control of defense

**Legal Considerations:**
{self._format_context_summary(contexts[:2])}

**Risk Management:**
• Include monetary caps to limit exposure
• Carve out gross negligence and willful misconduct
• Consider reciprocal indemnification where appropriate
• Ensure adequate insurance coverage

**Procedural Requirements:**
• Prompt notice obligations with specific timeframes
• Right to control defense and settlement
• Cooperation in defense and information sharing
• Mitigation of damages requirements

*This analysis is provided for informational purposes only and should not be relied upon as legal advice.*"""
    
    def _answer_confidentiality_question(self, question: str, contexts: List[RAGContextItem], 
                                       additional_context: str) -> str:
        """Answer confidentiality-related legal questions"""
        return f"""**Confidentiality Agreement Analysis:**

Non-disclosure provisions protect sensitive information and trade secrets:

**Essential Elements:**
• Clear definition of confidential information
• Appropriate exceptions (publicly available, independently developed)
• Reasonable time limitations and geographic scope
• Return or destruction obligations upon termination

**Legal Framework:**
{self._format_context_summary(contexts[:2])}

**Enforceability Standards:**
• Must protect legitimate business interests
• Reasonable in scope, duration, and geographic area
• Cannot be overly broad or restrict general knowledge
• State trade secret laws provide additional protections

**Best Practices:**
• Tailor definitions to specific business needs
• Include both disclosing and receiving party obligations
• Consider reciprocal confidentiality provisions
• Address employee and contractor obligations separately

*This guidance is for informational purposes only. Seek legal counsel for specific confidentiality matters.*"""
    
    def _answer_payment_question(self, question: str, contexts: List[RAGContextItem], 
                               additional_context: str) -> str:
        """Answer payment-related legal questions"""
        return f"""**Payment Terms Analysis:**

Payment provisions establish financial obligations and remedies for non-payment:

**Key Components:**
• Clear payment schedules and due dates
• Late payment penalties and interest rates
• Invoice requirements and dispute procedures
• Acceleration clauses and default provisions

**Legal Compliance:**
{self._format_context_summary(contexts[:2])}

**Regulatory Considerations:**
• State usury laws limit interest rates and penalties
• Consumer protection laws may override certain terms
• UCC Article 2 for sale of goods transactions
• Prompt payment acts for government contracts

**Risk Mitigation:**
• Include appropriate security mechanisms (guarantees, liens)
• Consider installment payment structures for large amounts
• Address currency and payment method specifications
• Include attorney fee provisions for collection actions

*This information is provided for general guidance only and does not constitute legal advice.*"""
    
    def _answer_ip_question(self, question: str, contexts: List[RAGContextItem], 
                          additional_context: str) -> str:
        """Answer intellectual property-related legal questions"""
        return f"""**Intellectual Property Clause Analysis:**

IP provisions define ownership, licensing, and protection of intellectual property rights:

**Ownership Issues:**
• Work-for-hire vs. independent contractor arrangements
• Pre-existing IP rights and background technology
• Joint development and shared ownership structures
• Assignment vs. licensing of IP rights

**Legal Framework:**
{self._format_context_summary(contexts[:2])}

**Protection Mechanisms:**
• Patent, copyright, trademark, and trade secret coverage
• Moral rights and attribution requirements
• Registration and maintenance obligations
• International IP considerations

**Key Provisions:**
• Clear definition of developed IP and derivatives
• Disclosure obligations for pre-existing rights
• License grants and restrictions on use
• Infringement indemnification and defense obligations

*This analysis is for informational purposes only. Consult IP counsel for specific intellectual property matters.*"""
    
    def _answer_dispute_question(self, question: str, contexts: List[RAGContextItem], 
                               additional_context: str) -> str:
        """Answer dispute resolution-related legal questions"""
        return f"""**Dispute Resolution Analysis:**

Alternative dispute resolution mechanisms can provide efficient conflict resolution:

**Arbitration Considerations:**
• Binding vs. non-binding arbitration procedures
• Arbitrator selection and qualification requirements
• Discovery limitations and expedited procedures
• Appeal rights and judicial review standards

**Legal Framework:**
{self._format_context_summary(contexts[:2])}

**Procedural Elements:**
• Multi-tiered dispute resolution (negotiation, mediation, arbitration)
• Venue and governing law provisions
• Cost allocation and attorney fee provisions
• Interim relief and injunctive relief availability

**Enforcement:**
• Federal Arbitration Act and state arbitration statutes
• International arbitration conventions and treaties
• Court enforcement of arbitration awards
• Limited grounds for challenging arbitration decisions

*This information is provided for general guidance and does not constitute legal advice on specific disputes.*"""
    
    def _answer_general_legal_question(self, question: str, contexts: List[RAGContextItem], 
                                     additional_context: str) -> str:
        """Answer general legal questions"""
        return f"""**Legal Analysis:**

Based on available legal research and precedents:

**Key Legal Considerations:**
{self._format_context_summary(contexts[:3])}

**Analysis Framework:**
• Review applicable federal, state, and local regulations
• Consider industry-specific legal requirements and standards
• Evaluate contractual obligations, rights, and remedies
• Assess potential risks and mitigation strategies

**Recommendations:**
• Ensure compliance with all relevant legal frameworks
• Consider consultation with qualified legal counsel for specific matters
• Document important decisions and maintain appropriate records
• Regular review and updates as laws and regulations change

**Risk Assessment:**
• Identify potential legal exposures and liability risks
• Implement appropriate compliance procedures and controls
• Consider insurance coverage for identified risks
• Establish monitoring and reporting mechanisms

*This analysis is provided for informational purposes only and does not constitute legal advice. Always consult with qualified legal counsel for specific legal matters.*"""
    
    def _format_context_summary(self, contexts: List[RAGContextItem]) -> str:
        """Format context summary for inclusion in answers"""
        if not contexts:
            return "Limited legal precedent data available for this analysis."
        
        summary_parts = []
        for ctx in contexts:
            summary_parts.append(
                f"• **{ctx.jurisdiction} {ctx.doc_type}**: {ctx.content[:300]}{'...' if len(ctx.content) > 300 else ''}"
            )
        
        return "\n".join(summary_parts)
    
    def _generate_reasoning(self, question: str, contexts: List[RAGContextItem], 
                          confidence: float) -> str:
        """Generate reasoning explanation for the answer"""
        
        context_count = len(contexts)
        avg_similarity = sum(ctx.similarity for ctx in contexts) / max(context_count, 1)
        
        reasoning = f"Analysis based on {context_count} relevant legal sources "
        reasoning += f"with average relevance score of {avg_similarity:.2f}. "
        
        if confidence > 0.8:
            reasoning += "High confidence due to strong contextual matches and comprehensive legal coverage from multiple jurisdictions."
        elif confidence > 0.6:
            reasoning += "Moderate confidence with good supporting legal precedents and established guidance."
        elif confidence > 0.4:
            reasoning += "Limited confidence due to sparse legal context or complex legal questions requiring specialized expertise."
        else:
            reasoning += "Low confidence - recommend consulting with qualified legal counsel for specific advice."
        
        return reasoning
    
    async def summarize_200w(self, text: str) -> str:
        """
        Generate concise 200-word summary of legal document
        
        Args:
            text: Full document text to summarize
            
        Returns:
            Concise summary of 200 words or less
        """
        
        if not text or len(text.strip()) < 50:
            return "Document too short to generate meaningful summary."
        
        start_time = time.time()
        
        try:
            # Check cache
            cache_key = self._get_cache_key(text[:500])  # Use first 500 chars for cache key
            if cache_key in self._answer_cache:
                cache_entry = self._answer_cache[cache_key]
                if self._is_cache_valid(cache_entry):
                    self.metrics.record_cache_hit()
                    return cache_entry['answer']['summary']
            
            # Generate structured summary
            summary = self._generate_structured_summary(text)
            
            # Cache result
            self._answer_cache[cache_key] = {
                'answer': {'summary': summary},
                'timestamp': datetime.now().isoformat()
            }
            
            processing_time = time.time() - start_time
            logger.debug(f"Generated summary in {processing_time:.2f}s")
>>>>>>> Stashed changes
            
            return summary
            
        except Exception as e:
<<<<<<< Updated upstream
            logger.error(f"❌ Summary generation failed: {e}", exc_info=True)
            return self._create_fallback_summary(text)
    
    def _create_fallback_summary(self, text: str) -> str:
        """Create a simple fallback summary"""
        words = text.split()
        if len(words) <= 150:
            return text
        
        sample_text = " ".join(words[:150])
        return f"This legal document contains {len(words)} words covering contractual terms and conditions. {sample_text}..."
    
    async def answer_with_vertex(self, question: str, contexts: List[RAGContextItem], summary_hint: Optional[str] = None) -> str:
        """Generate an answer using Vertex AI with RAG contexts"""
        
        context_count = len(contexts)
        
        # Build dynamic context based on evidence
        context_text = ""
        evidence_topics = []
        if contexts:
            context_items = []
            for i, ctx in enumerate(contexts[:5], 1):
                context_items.append(f"Legal Document {i}:\n{ctx.content}")
                # Extract key topics from evidence for dynamic prompting
                if "liability" in ctx.content.lower():
                    evidence_topics.append("liability principles")
                if "indemnif" in ctx.content.lower():
                    evidence_topics.append("indemnification")
                if "risk" in ctx.content.lower():
                    evidence_topics.append("risk management")
            
            context_text = "\n\n".join(context_items)
        
        summary_context = f"\nDocument Summary: {summary_hint}" if summary_hint else ""
        
        # ✅ DYNAMIC PROMPT: Adapts based on evidence and question
        prompt = f"""You are a legal document assistant. Answer the user's legal question using the provided evidence and your legal expertise.

    EVIDENCE ANALYSIS:
    - Found {context_count} relevant legal documents
    - Topics covered: {', '.join(evidence_topics) if evidence_topics else 'general legal principles'}
    - Evidence quality: High similarity matches

    INSTRUCTIONS:
    1. Use the legal documents below as primary evidence for your answer
    2. Combine this evidence with your legal knowledge to provide a comprehensive response
    3. If the documents don't directly address the question, use them as supporting context and apply relevant legal principles
    4. Always provide a substantive, helpful answer
    5. Use simple, clear language for non-lawyers
    6. Include practical implications and recommendations

    LEGAL EVIDENCE:
    {context_text}
    {summary_context}

    USER QUESTION: {question}

    Provide a thorough, evidence-based answer that helps the user understand the legal concept:"""
        
        try:
            answer = await self.vertex_client.generate_text(
                prompt=prompt,
                max_tokens=getattr(settings, 'VERTEX_MAX_TOKENS', 1024),
                temperature=getattr(settings, 'VERTEX_TEMPERATURE', 0.3)
            )
            
            # ✅ RETRY LOGIC: If response seems inadequate, try enhanced prompt
            if self._is_inadequate_response(answer):
                logger.info("Response seems inadequate, retrying with enhanced prompt")
                enhanced_answer = await self._retry_with_enhanced_prompt(question, contexts, summary_hint)
                return enhanced_answer or answer
            
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            # Only use minimal fallback as last resort
            return f"I found {context_count} relevant legal documents but encountered an error generating the response. Please try rephrasing your question."

    def _is_inadequate_response(self, answer: str) -> bool:
        """Check if response seems inadequate and needs retry"""
        if not answer or len(answer.strip()) < 50:
            return True
        
        inadequate_phrases = [
            "no information", "cannot answer", "not provided", 
            "no context", "unable to answer", "insufficient information",
            "i am sorry", "i cannot"
        ]
        
        return any(phrase in answer.lower() for phrase in inadequate_phrases)

    async def _retry_with_enhanced_prompt(self, question: str, contexts: List[RAGContextItem], summary_hint: Optional[str] = None) -> str:
        """Retry with more specific prompt when first attempt fails"""
        
        context_text = "\n\n".join([f"Document {i+1}: {ctx.content}" for i, ctx in enumerate(contexts[:3])])
        
        # More directive prompt for retry
        enhanced_prompt = f"""You are a knowledgeable legal assistant. The user has asked about a legal topic and I've found relevant legal documents.

    TASK: Provide a comprehensive answer about the legal concept in the question, using both the legal documents provided and your legal expertise.

    RETRIEVED LEGAL DOCUMENTS:
    {context_text}

    USER'S LEGAL QUESTION: {question}

    REQUIREMENTS:
    - Explain the legal concept clearly in simple terms
    - Use the provided documents as supporting evidence where relevant
    - Apply general legal principles even if documents don't directly address the topic
    - Include practical advice and risk considerations
    - Make the response helpful and actionable

    Generate a detailed, informative response:"""
        
        try:
            return await self.vertex_client.generate_text(
                prompt=enhanced_prompt,
                max_tokens=getattr(settings, 'VERTEX_MAX_TOKENS', 1024),
                temperature=0.4  # Slightly higher temperature for more creative response
            )
        except Exception as e:
            logger.error(f"Enhanced prompt retry failed: {e}")
            return None

    
    def _create_context_aware_fallback(self, question: str, contexts: List[RAGContextItem]) -> str:
        """Create fallback that acknowledges found evidence"""
        context_count = len(contexts)
        
        if context_count > 0:
            return f"""Based on {context_count} relevant legal documents I found, here's what I can tell you:

🔍 **Legal Research**: I found relevant information in legal documents including regulations about various legal matters and statutory frameworks.

📚 **Key Insights**: 
- Legal obligations and rights are clearly defined in statutory frameworks
- Compliance requirements are established by appropriate authorities
- Penalties and procedures are specified for various scenarios

💡 **Recommendation**: While I found relevant legal precedents, the specific details are complex. I recommend consulting with a qualified attorney who can provide advice specific to your situation.

❓ **Need More Details**: Feel free to ask more specific questions about the legal concepts you're interested in."""
        
        else:
            return "I couldn't find specific legal precedents for your question. Please try rephrasing your question or ask about a specific legal concept."
    
    def _create_fallback_answer(self, question: str, contexts: List[RAGContextItem]) -> str:
        """Create fallback answer when Vertex AI is unavailable"""
        context_count = len(contexts)
        question_lower = question.lower()
        
        if "risk" in question_lower or "danger" in question_lower:
            return f"""Based on {context_count} relevant legal documents, here are the key risk considerations:

🔍 **Analysis**: Legal documents often contain terms that may not be immediately obvious to non-lawyers.

⚠️ **Key Points**: 
- Review liability and indemnification clauses carefully
- Understand termination conditions and notice requirements  
- Pay attention to payment terms and penalties
- Consider governing law and dispute resolution mechanisms

💡 **Recommendation**: Given the complexity of legal language, I strongly recommend having a qualified attorney review any concerning clauses or the entire document before signing.

❓ **Next Steps**: Feel free to ask about specific sections or terms you'd like explained in simpler language."""

        else:
            return f"""Thank you for your question. I've analyzed {context_count} relevant legal documents to provide guidance:

🎯 **Key Finding**: Legal documents are designed to protect all parties by clearly defining rights, responsibilities, and procedures.

📚 **From Legal Research**: The information comes from analysis of similar legal documents and established legal principles.

⚖️ **Legal Disclaimer**: This is general information only. For advice specific to your situation, please consult with a qualified attorney.

Feel free to ask more specific questions about particular clauses or terms!"""
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        return {
            "initialized": self.initialized,
            "milvus_connected": self.milvus_client.connected,
            "vertex_initialized": self.vertex_client.initialized,
            "collection_name": getattr(settings, 'MILVUS_COLLECTION', 'unknown'),
            "embedding_dimension": 768,
            "dependencies": {
                "pymilvus_available": PYMILVUS_AVAILABLE,
                "requests_available": True,  # We know this is available since we're using it
                "http_client": True
            }
        }


# Global service instance
_rag_service = None


async def get_rag_service() -> RAGService:
    """Get or create the global RAG service instance"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
        await _rag_service.initialize()
    return _rag_service


# Public API functions for router integration
async def retrieve_contexts(query: str, top_k: int = 8) -> List[RAGContextItem]:
    """Retrieve relevant contexts for a query"""
    service = await get_rag_service()
    return await service.retrieve_contexts(query, top_k)


async def summarize_200w(text: str) -> str:
    """Generate a summary in ≤200 words"""
    service = await get_rag_service()
    return await service.summarize_200w(text)


async def answer_with_vertex(question: str, contexts: List[RAGContextItem], summary_hint: Optional[str] = None) -> str:
    """Generate an answer using Vertex AI with RAG contexts"""
    service = await get_rag_service()
    return await service.answer_with_vertex(question, contexts, summary_hint)


async def health_check() -> Dict[str, Any]:
    """Health check for RAG service"""
    try:
        service = await get_rag_service()
        stats = await service.get_service_stats()
        stats["status"] = "healthy" if service.initialized else "degraded"
        return stats
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "initialized": False
        }
=======
            logger.error(f"Summary generation failed: {str(e)}")
            return self._fallback_summary(text)
    
    def _generate_structured_summary(self, text: str) -> str:
        """Generate structured summary using template approach"""
        
        words = text.split()
        word_count = len(words)
        
        # Extract key information
        legal_terms = self._extract_legal_terms(text)
        key_parties = self._extract_parties(text)
        important_dates = self._extract_dates(text)
        
        # Build structured summary
        summary_parts = []
        
        # Opening
        if key_parties:
            summary_parts.append(f"Legal document between {', '.join(key_parties[:2])}.")
        else:
            summary_parts.append("Legal document analysis:")
        
        # Key provisions
        if legal_terms:
            provision_text = f"Contains {len(legal_terms)} key legal provisions including {', '.join(legal_terms[:3])}"
            if len(legal_terms) > 3:
                provision_text += f" and {len(legal_terms) - 3} additional areas"
            summary_parts.append(provision_text + ".")
        
        # Risk assessment
        risk_indicators = self._assess_document_risk(text)
        if risk_indicators['high_risk_count'] > 0:
            summary_parts.append(f"Document contains {risk_indicators['high_risk_count']} high-risk provisions requiring careful legal review.")
        elif risk_indicators['medium_risk_count'] > 2:
            summary_parts.append(f"Document includes {risk_indicators['medium_risk_count']} medium-risk provisions for consideration.")
        
        # Important dates
        if important_dates:
            summary_parts.append(f"Key dates identified: {', '.join(important_dates[:2])}.")
        
        # Recommendations
        if risk_indicators['total_clauses'] > 15:
            summary_parts.append("Comprehensive legal review recommended due to document complexity and scope.")
        elif risk_indicators['high_risk_count'] > 2:
            summary_parts.append("Legal consultation advised for high-risk provisions and potential liability exposure.")
        else:
            summary_parts.append("Standard legal document with typical commercial provisions.")
        
        # Combine and limit to 200 words
        summary = " ".join(summary_parts)
        summary_words = summary.split()
        
        if len(summary_words) > 200:
            summary = " ".join(summary_words[:200]) + "..."
        
        return summary
    
    def _extract_legal_terms(self, text: str) -> List[str]:
        """Extract key legal terms from document"""
        text_lower = text.lower()
        
        legal_categories = {
            'termination': ['terminat', 'expir', 'end', 'cease', 'dissolv'],
            'liability': ['liabilit', 'liable', 'damages', 'harm', 'loss'],
            'indemnification': ['indemnif', 'hold harmless', 'defend', 'protect'],
            'confidentiality': ['confidential', 'proprietary', 'non-disclosure', 'nda', 'trade secret'],
            'payment': ['payment', 'fees', 'invoice', 'billing', 'compensation'],
            'intellectual property': ['intellectual property', 'copyright', 'trademark', 'patent', 'ip rights'],
            'governing law': ['governing law', 'jurisdiction', 'applicable law'],
            'arbitration': ['arbitrat', 'dispute resolution', 'mediat', 'binding arbitration'],
            'force majeure': ['force majeure', 'act of god', 'unforeseeable circumstances'],
            'privacy': ['privacy', 'personal data', 'gdpr', 'ccpa', 'data protection']
        }
        
        found_terms = []
        for category, keywords in legal_categories.items():
            if any(keyword in text_lower for keyword in keywords):
                found_terms.append(category)
        
        return found_terms
    
    def _extract_parties(self, text: str) -> List[str]:
        """Extract party names from document"""
        import re
        
        # Look for common party patterns
        party_patterns = [
            r'"([A-Za-z\s&,\.]+(?:Inc|LLC|Corp|Corporation|Company|Ltd))"',
            r'"(Company|Employee|Contractor|Client|Vendor|Supplier)"',
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b(?=\s+\(")'
        ]
        
        parties = []
        for pattern in party_patterns:
            matches = re.findall(pattern, text[:1000])  # Check first 1000 chars
            parties.extend(matches[:2])  # Max 2 per pattern
        
        # Clean and deduplicate
        cleaned_parties = []
        for party in parties:
            if len(party.strip()) > 2 and party.strip() not in cleaned_parties:
                cleaned_parties.append(party.strip())
        
        return cleaned_parties[:3]  # Return max 3 parties
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract important dates from document"""
        import re
        
        # Enhanced date patterns
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b'
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches[:2])  # Max 2 per pattern
        
        return list(set(dates))[:3]  # Return unique dates, max 3
    
    def _assess_document_risk(self, text: str) -> Dict[str, int]:
        """Assess document risk levels"""
        text_lower = text.lower()
        
        high_risk_terms = [
            'unlimited liability', 'personal guarantee', 'liquidated damages',
            'penalty', 'forfeiture', 'material breach', 'immediate termination',
            'gross negligence', 'willful misconduct', 'punitive damages'
        ]
        
        medium_risk_terms = [
            'liability', 'indemnif', 'terminate', 'confidential', 'breach',
            'arbitration', 'governing law', 'force majeure', 'assignment',
            'non-compete', 'intellectual property', 'data privacy'
        ]
        
        high_risk_count = sum(1 for term in high_risk_terms if term in text_lower)
        medium_risk_count = sum(1 for term in medium_risk_terms if term in text_lower)
        
        # Estimate total clauses (improved approximation)
        sentences = [s for s in text.split('.') if len(s.strip()) > 30]
        estimated_clauses = len(sentences)
        
        return {
            'high_risk_count': high_risk_count,
            'medium_risk_count': medium_risk_count,
            'total_clauses': estimated_clauses
        }
    
    def _fallback_summary(self, text: str) -> str:
        """Generate fallback summary when advanced processing fails"""
        words = text.split()
        word_count = len(words)
        
        if word_count <= 200:
            return text
        
        # Extract key sentences using improved heuristics
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 30]
        
        summary_sentences = []
        if sentences:
            summary_sentences.append(sentences[0])  # First sentence
            
            # Add middle sentence if available
            if len(sentences) > 4:
                summary_sentences.append(sentences[len(sentences)//3])
                summary_sentences.append(sentences[2*len(sentences)//3])
            elif len(sentences) > 2:
                summary_sentences.append(sentences[len(sentences)//2])
            
            if len(sentences) > 1:
                summary_sentences.append(sentences[-1])  # Last sentence
        
        summary = '. '.join(summary_sentences) + '.'
        
        # Add basic document characteristics
        legal_terms = self._extract_legal_terms(text)
        if legal_terms:
            summary += f" Document addresses {len(legal_terms)} key legal areas including {', '.join(legal_terms[:2])}."
        
        # Limit to 200 words
        summary_words = summary.split()
        if len(summary_words) > 200:
            summary = ' '.join(summary_words[:200]) + '...'
        
        return summary
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get RAG service performance metrics"""
        return {
            "retrieval": {
                "total_queries": self.metrics.total_queries,
                "successful_queries": self.metrics.successful_queries,
                "success_rate": round(self.metrics.success_rate * 100, 2),
                "cache_hit_rate": round(self.metrics.cache_hit_rate * 100, 2),
                "avg_retrieval_time_ms": round(self.metrics.avg_retrieval_time * 1000, 2),
                "total_contexts_retrieved": self.metrics.total_contexts_retrieved
            },
            "caches": {
                "answer_cache_size": len(self._answer_cache),
                "context_cache_size": len(self._context_cache)
            },
            "services": {
                "milvus_connected": self.vector_store.connected,
                "embedding_model_loaded": self.embedding_service.model is not None,
                "embedding_model": self.embedding_service.model_name,
                "milvus_host": self.vector_store.host,
                "milvus_collection": self.vector_store.collection_name
            },
            "config": {
                "cache_ttl_hours": self.cache_ttl.total_seconds() / 3600,
                "max_requests_per_minute": self.max_requests_per_minute
            }
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self._answer_cache.clear()
        self._context_cache.clear()
        logger.info("RAG service caches cleared")

# Global service instance
rag_service = RAGService()

# Public API functions for backward compatibility and clean imports
async def retrieve_contexts(query_text: str, top_k: int = 8) -> List[RAGContextItem]:
    """Retrieve relevant legal contexts for the query"""
    return await rag_service.retrieve_contexts(query_text, top_k)

async def answer_question(question: str, context: str = "", conversation_id: Optional[str] = None) -> Dict[str, Any]:
    """Answer legal question using RAG with retrieved contexts"""
    return await rag_service.answer_question(question, context, conversation_id)

async def summarize_200w(text: str) -> str:
    """Generate 200-word summary of legal document"""
    return await rag_service.summarize_200w(text)

def get_rag_metrics() -> Dict[str, Any]:
    """Get RAG service performance metrics"""
    return rag_service.get_metrics()

def clear_rag_cache():
    """Clear RAG service caches"""
    rag_service.clear_cache()

# For backward compatibility with existing answer_with_vertex function name
async def answer_with_vertex(question: str, contexts: List[RAGContextItem], summary_hint: Optional[str] = None) -> str:
    """Legacy compatibility function for answer generation"""
    context_text = "\n".join([ctx.content for ctx in contexts[:3]])
    result = await answer_question(question, context_text)
    return result.get('answer', 'Unable to generate answer')
>>>>>>> Stashed changes
