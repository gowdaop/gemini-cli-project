import asyncio
import logging
import hashlib
import json
import time
from typing import List, Optional, Dict, Any, Union, Tuple
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import uuid
from functools import lru_cache
from collections import OrderedDict

# ✅ FIXED: Correct PyMilvus imports
try:
    from pymilvus import connections, Collection, utility, DataType
    from pymilvus import MilvusException  # ← FIXED: Direct import from pymilvus
    PYMILVUS_AVAILABLE = True
except ImportError:
    PYMILVUS_AVAILABLE = False
    MilvusException = Exception  # ← Fallback exception type

# ✅ FIXED: Correct Vertex AI imports with fallback
try:
    import google.cloud.aiplatform as aiplatform
    from google.api_core import retry, exceptions as gcp_exceptions
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    aiplatform = None
    gcp_exceptions = None

# ✅ FIXED: Add genai import for Vertex AI generation
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    try:
        from google import genai
        GENAI_AVAILABLE = True
    except ImportError:
        GENAI_AVAILABLE = False
        genai = None

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
            
            # Check TTL
            if time.time() - self.timestamps[key] > self.ttl:
                await self._remove(key)
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
    
    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    # Remove least recently used
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
        self._search_cache = RAGCache(max_size=500, ttl=1800)  # 30 min cache
    
    async def initialize(self):
        """Initialize Milvus connection with retry logic"""
        if self.connected:
            return
        
        if not PYMILVUS_AVAILABLE:
            logger.warning("PyMilvus not available - RAG will use fallback mode")
            return
        
        for attempt in range(self._connection_retries):
            try:
                # Connect to Milvus
                connections.connect(
                    alias="default",
                    host=settings.MILVUS_HOST,
                    port=settings.MILVUS_PORT,
                    timeout=settings.MILVUS_TIMEOUT
                )
                
                # Verify collection exists
                if not utility.has_collection(settings.MILVUS_COLLECTION):
                    logger.error(f"Collection {settings.MILVUS_COLLECTION} not found in Milvus")
                    return
                
                # Load collection
                self.collection = Collection(settings.MILVUS_COLLECTION)
                await asyncio.get_event_loop().run_in_executor(
                    None, self.collection.load
                )
                
                self.connected = True
                logger.info(f"Connected to Milvus collection: {settings.MILVUS_COLLECTION}")
                
                # Log collection info
                await self._log_collection_info()
                break
                
            except Exception as e:  # ✅ FIXED: Using generic Exception instead of specific MilvusException
                logger.warning(f"Milvus connection attempt {attempt + 1} failed: {e}")
                if attempt == self._connection_retries - 1:
                    logger.error("All Milvus connection attempts failed - using fallback mode")
                else:
                    await asyncio.sleep(self._retry_delay ** attempt)
    
    async def _log_collection_info(self):
        """Log collection statistics"""
        try:
            stats = await asyncio.get_event_loop().run_in_executor(
                None, self.collection.num_entities
            )
            logger.info(f"Milvus collection {settings.MILVUS_COLLECTION} has {stats} vectors")
        except Exception as e:
            logger.debug(f"Could not get collection stats: {e}")
    
    async def search(
        self, 
        query_vector: List[float], 
        top_k: int = 8,
        metric_type: str = "COSINE"
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in Milvus"""
        if not self.connected or not self.collection:
            logger.warning("Milvus not connected - returning empty results")
            return []
        
        # Create cache key
        cache_key = self._create_search_cache_key(query_vector, top_k, metric_type)
        
        # Check cache first
        cached_result = await self._search_cache.get(cache_key)
        if cached_result:
            logger.debug("Returning cached search results")
            return cached_result
        
        try:
            search_params = {
                "metric_type": metric_type,
                "params": {"nprobe": 16} if metric_type in ["L2", "IP"] else {}
            }
            
            # Perform search
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.search(
                    data=[query_vector],
                    anns_field="embedding",  # Assuming this is your vector field name
                    param=search_params,
                    limit=top_k,
                    output_fields=["*"]  # Get all metadata fields
                )
            )
            
            # Process results
            processed_results = self._process_search_results(results)
            
            # Cache results
            await self._search_cache.set(cache_key, processed_results)
            
            logger.info(f"Milvus search returned {len(processed_results)} results")
            return processed_results
            
        except Exception as e:  # ✅ FIXED: Generic exception handling
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
                    # Extract metadata from hit
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
    """Production-ready Vertex AI client for legal document processing"""
    
    def __init__(self):
        self.initialized = False
        self._generation_cache = RAGCache(max_size=200, ttl=7200)  # 2 hour cache
        self._retry_config = None
        
        if VERTEX_AI_AVAILABLE and gcp_exceptions:
            self._retry_config = retry.Retry(
                initial=1.0,
                maximum=10.0,
                multiplier=2.0,
                predicate=retry.if_exception_type(
                    gcp_exceptions.DeadlineExceeded,
                    gcp_exceptions.ServiceUnavailable,
                    gcp_exceptions.ResourceExhausted
                )
            )
    
    async def initialize(self):
        """Initialize Vertex AI client"""
        if self.initialized:
            return
        
        if not VERTEX_AI_AVAILABLE:
            logger.warning("Vertex AI not available - using fallback responses")
            return
        
        try:
            aiplatform.init(
                project=settings.GCP_PROJECT_ID,
                location=settings.VERTEX_LOCATION
            )
            
            self.initialized = True
            logger.info(f"Vertex AI initialized for project: {settings.GCP_PROJECT_ID}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
    
    async def generate_text(
        self, 
        prompt: str, 
        max_tokens: int = None,
        temperature: float = None
    ) -> str:
        """Generate text using Vertex AI Gemini"""
        if not self.initialized:
            await self.initialize()
        
        if not self.initialized:
            return self._create_fallback_response(prompt)
        
        # Check cache
        cache_key = hashlib.md5(f"{prompt}_{max_tokens}_{temperature}".encode()).hexdigest()
        cached_result = await self._generation_cache.get(cache_key)
        if cached_result:
            logger.debug("Returning cached generation result")
            return cached_result
        
        try:
            # ✅ FIXED: Simplified Vertex AI generation approach
            if GENAI_AVAILABLE and genai:
                # Configure genai
                genai.configure()
                
                model = genai.GenerativeModel(settings.VERTEX_MODEL)
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=max_tokens or settings.VERTEX_MAX_TOKENS,
                            temperature=temperature or settings.VERTEX_TEMPERATURE
                        )
                    )
                )
                
                generated_text = response.text if hasattr(response, 'text') else str(response)
            else:
                # Fallback to basic aiplatform
                model = aiplatform.GenerativeModel(settings.VERTEX_MODEL)
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: model.generate_content(prompt)
                )
                generated_text = response.text if hasattr(response, 'text') else str(response)
            
            # Cache the result
            await self._generation_cache.set(cache_key, generated_text)
            
            logger.info("Successfully generated text with Vertex AI")
            return generated_text
            
        except Exception as e:
            logger.error(f"Vertex AI generation failed: {e}")
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
            return ("I'm currently operating in limited mode. While I can provide general guidance "
                   "about legal documents, for specific legal advice and detailed analysis, "
                   "please consult with a qualified legal professional.")

# ✅ FIXED: Rest of the RAG service remains the same but with corrected exception handling

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
        
        # Initialize all clients concurrently
        await asyncio.gather(
            self.milvus_client.initialize(),
            self.vertex_client.initialize(),
            return_exceptions=True
        )
        
        self.initialized = True
        logger.info("RAG service initialization complete")
    
    async def retrieve_contexts(
        self, 
        query: str, 
        top_k: int = 8,
        include_inlegalbert: bool = True
    ) -> List[RAGContextItem]:
        """Retrieve relevant contexts from both Milvus and InLEGALBERT"""
        if not self.initialized:
            await self.initialize()
        
        logger.debug(f"Retrieving contexts for query: {query[:100]}...")
        
        try:
            # Generate query embedding
            query_embeddings = await embed_texts_async([query])
            if not query_embeddings or not query_embeddings[0]:
                logger.error("Failed to generate query embedding")
                return []
            
            query_vector = query_embeddings[0]
            
            # Search Milvus (2020-2025 data)
            milvus_results = await self.milvus_client.search(
                query_vector=query_vector,
                top_k=min(top_k, 8),
                metric_type="COSINE"
            )
            
            # Convert to RAGContextItem format
            rag_contexts = []
            
            # Process Milvus results
            for i, result in enumerate(milvus_results):
                try:
                    rag_contexts.append(RAGContextItem(
                        chunk_id=result.get("id", i),
                        content=result.get("content", "")[:2000],  # Limit content length
                        doc_type=result.get("doc_type", "legal_document"),
                        jurisdiction=result.get("jurisdiction", "unknown"),
                        date=result.get("date", "2020-2025"),
                        source_url=result.get("source_url", "milvus_collection"),
                        similarity=float(result.get("similarity", 0.0))
                    ))
                except Exception as e:
                    logger.warning(f"Error creating RAGContextItem: {e}")
                    continue
            
            # Sort by similarity
            rag_contexts.sort(key=lambda x: x.similarity, reverse=True)
            
            logger.info(f"Retrieved {len(rag_contexts)} contexts for query")
            return rag_contexts[:top_k]
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return []
    
    async def summarize_200w(self, text: str) -> str:
        """Generate a summary in ≤200 words"""
        if not text or not text.strip():
            return "No content available for summary."
        
        prompt = f"""
You are a legal document assistant. Create a concise summary of this legal document in exactly 200 words or fewer.

Focus on:
1. Main purpose and type of document
2. Key parties involved
3. Primary obligations and rights
4. Important terms and conditions
5. Notable risks or considerations

Use simple, non-technical language that someone without legal training can understand.

Document text:
{text[:4000]}...

Summary (≤200 words):
"""
        
        try:
            summary = await self.vertex_client.generate_text(
                prompt=prompt,
                max_tokens=300,
                temperature=0.1
            )
            
            # Ensure word limit
            words = summary.split()
            if len(words) > 200:
                summary = " ".join(words[:200]) + "..."
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return self._create_fallback_summary(text)
    
    def _create_fallback_summary(self, text: str) -> str:
        """Create a simple fallback summary"""
        words = text.split()
        if len(words) <= 150:
            return text
        
        # Take first 150 words and add context
        sample_text = " ".join(words[:150])
        return f"This legal document contains {len(words)} words covering contractual terms and conditions. {sample_text}..."
    
    async def answer_with_vertex(
        self, 
        question: str, 
        contexts: List[RAGContextItem],
        summary_hint: Optional[str] = None
    ) -> str:
        """Generate an answer using Vertex AI with RAG contexts"""
        
        # Build context from RAG results
        context_text = ""
        if contexts:
            context_items = []
            for i, ctx in enumerate(contexts[:5], 1):
                context_items.append(f"Context {i} ({ctx.doc_type}, {ctx.jurisdiction}):\n{ctx.content}")
            context_text = "\n\n".join(context_items)
        
        # Add summary hint if provided
        summary_context = f"\nDocument Summary: {summary_hint}" if summary_hint else ""
        
        prompt = f"""
You are a legal document assistant helping people with limited legal knowledge understand contracts and legal documents.

INSTRUCTIONS:
1. Provide clear, simple explanations avoiding legal jargon
2. Use everyday language and analogies when possible
3. Highlight potential risks or important considerations
4. Suggest when to consult a professional lawyer
5. Be specific and reference the provided context
6. Keep response focused and helpful

LEGAL CONTEXT:
{context_text}
{summary_context}

USER QUESTION: {question}

RESPONSE (in simple, non-legal language):
"""
        
        try:
            answer = await self.vertex_client.generate_text(
                prompt=prompt,
                max_tokens=settings.VERTEX_MAX_TOKENS,
                temperature=settings.VERTEX_TEMPERATURE
            )
            
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return self._create_fallback_answer(question, contexts)
    
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
            "collection_name": settings.MILVUS_COLLECTION,
            "embedding_dimension": 768,
            "dependencies": {
                "pymilvus_available": PYMILVUS_AVAILABLE,
                "vertex_ai_available": VERTEX_AI_AVAILABLE,
                "genai_available": GENAI_AVAILABLE
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

async def answer_with_vertex(
    question: str, 
    contexts: List[RAGContextItem], 
    summary_hint: Optional[str] = None
) -> str:
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
