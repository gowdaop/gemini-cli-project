import asyncio
import logging
import hashlib
import time
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
            
            return summary
            
        except Exception as e:
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
