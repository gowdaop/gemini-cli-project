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

# PyMilvus imports
try:
    from pymilvus import connections, Collection, utility, DataType
    from pymilvus import MilvusException
    PYMILVUS_AVAILABLE = True
except ImportError:
    PYMILVUS_AVAILABLE = False
    MilvusException = Exception

# Google Custom Search imports (legacy - keep for compatibility)
try:
    from googleapiclient.discovery import build
    GOOGLE_SEARCH_AVAILABLE = True
except ImportError:
    GOOGLE_SEARCH_AVAILABLE = False

# Vertex AI Search imports (new)
try:
    from google.cloud import discoveryengine_v1beta as discoveryengine
    VERTEX_SEARCH_AVAILABLE = True
except ImportError:
    VERTEX_SEARCH_AVAILABLE = False
    discoveryengine = None

# Free web search imports
try:
    import aiohttp
    ASYNC_HTTP_AVAILABLE = True
except ImportError:
    ASYNC_HTTP_AVAILABLE = False

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


class VertexAISearchService:
    """Vertex AI Search service for Indian legal content"""
    
    def __init__(self):
        self.project_id = "legal-470717"  # Your project ID
        self.location = "global"
        self.engine_id = "legal-document-search_1757848376569"  # Your engine ID
        self.client = None
        self.initialized = False
        self._search_cache = RAGCache(max_size=200, ttl=3600)
        
    async def initialize(self):
        """Initialize Vertex AI Search service"""
        if self.initialized:
            return
            
        if not VERTEX_SEARCH_AVAILABLE:
            logger.warning("Vertex AI Search not available - web search disabled")
            return
            
        try:
            self.client = discoveryengine.SearchServiceClient()
            self.initialized = True
            logger.info("✅ Vertex AI Search service initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Vertex AI Search: {e}")
    
    async def search_web(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search using Vertex AI Search for Indian legal content"""
        if not self.initialized:
            await self.initialize()
            
        if not self.initialized:
            logger.warning("Vertex AI Search not available - returning empty results")
            return []
            
        # Check cache first
        cache_key = hashlib.md5(f"vertex_web_{query}_{num_results}".encode()).hexdigest()
        cached_result = await self._search_cache.get(cache_key)
        if cached_result:
            logger.debug("Returning cached Vertex AI search results")
            return cached_result
            
        try:
            logger.info(f"🔍 Vertex AI Search for: '{query[:50]}...'")
            
            serving_config = (
                f"projects/{self.project_id}/locations/{self.location}/"
                f"collections/default_collection/engines/{self.engine_id}/servingConfigs/default_search"
            )
            
            request = discoveryengine.SearchRequest(
                serving_config=serving_config,
                query=query,
                page_size=min(num_results, 10),
                content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(
                    snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                        return_snippet=True
                    )
                )
            )
            
            # Execute search
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.client.search(request)
            )
            
            search_results = []
            for result in response.results:
                try:
                    document = result.document
                    
                    # Extract title and content
                    title = document.derived_struct_data.get('title', '') if document.derived_struct_data else ''
                    link = document.derived_struct_data.get('link', '') if document.derived_struct_data else ''
                    
                    # Get snippet from search result
                    snippet = ""
                    if hasattr(result, 'document') and hasattr(result.document, 'derived_struct_data'):
                        snippets = result.document.derived_struct_data.get('snippets', [])
                        if snippets:
                            snippet = snippets[0].get('snippet', '')
                    
                    search_results.append({
                        'title': title,
                        'link': link,
                        'snippet': snippet,
                        'displayLink': link.split('/')[2] if '//' in link else link,
                        'source': 'vertex_ai_search'
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing Vertex AI result: {e}")
                    continue
            
            # Cache results
            await self._search_cache.set(cache_key, search_results)
            
            logger.info(f"✅ Vertex AI Search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"❌ Vertex AI Search failed: {e}")
            return []
    
    def convert_to_rag_context(self, web_results: List[Dict[str, Any]]) -> List[RAGContextItem]:
        """Convert Vertex AI search results to RAG context items"""
        rag_contexts = []
        
        for i, result in enumerate(web_results):
            try:
                # Create content from title and snippet
                content = f"{result.get('title', '')}. {result.get('snippet', '')}"
                
                rag_contexts.append(RAGContextItem(
                    chunk_id=i,
                    content=content[:2000],  # Limit content length
                    doc_type="vertex_ai_search_result",
                    jurisdiction="indian_law",
                    date=datetime.now().isoformat()[:10],
                    source_url=result.get('link', ''),
                    similarity=0.85  # High similarity for Vertex AI results
                ))
            except Exception as e:
                logger.warning(f"❌ Error converting Vertex AI result {i}: {e}")
                continue
        
        return rag_contexts


class FreeWebSearchService:
    """Free web search service using DuckDuckGo API as fallback"""
    
    def __init__(self):
        self.base_url = "https://api.duckduckgo.com"
        self.initialized = False
        self._search_cache = RAGCache(max_size=200, ttl=1800)
        
    async def initialize(self):
        """Initialize free web search service"""
        if self.initialized:
            return
            
        if not ASYNC_HTTP_AVAILABLE:
            logger.warning("aiohttp not available - free web search disabled")
            return
            
        try:
            self.initialized = True
            logger.info("✅ Free web search service initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize free web search: {e}")
    
    async def search_web(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo API for free web search"""
        if not self.initialized:
            await self.initialize()
            
        if not self.initialized:
            logger.warning("Free web search not available - returning empty results")
            return []
            
        # Check cache first
        cache_key = hashlib.md5(f"free_web_{query}_{num_results}".encode()).hexdigest()
        cached_result = await self._search_cache.get(cache_key)
        if cached_result:
            logger.debug("Returning cached free web search results")
            return cached_result
            
        try:
            logger.info(f"🔍 Free web search for: '{query[:50]}...'")
            
            # Use DuckDuckGo Instant Answer API
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        search_results = self._process_duckduckgo_results(data, num_results)
                        
                        # Cache results
                        await self._search_cache.set(cache_key, search_results)
                        
                        logger.info(f"✅ Free web search returned {len(search_results)} results")
                        return search_results
                    else:
                        logger.warning(f"Free web search failed with status {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"❌ Free web search failed: {e}")
            return []
    
    def _process_duckduckgo_results(self, data: Dict, num_results: int) -> List[Dict[str, Any]]:
        """Process DuckDuckGo API results"""
        search_results = []
        
        # Extract abstract and related topics
        if data.get('Abstract'):
            search_results.append({
                'title': data.get('Heading', 'DuckDuckGo Result'),
                'link': data.get('AbstractURL', ''),
                'snippet': data.get('Abstract', ''),
                'displayLink': 'duckduckgo.com',
                'source': 'duckduckgo_instant'
            })
        
        # Extract related topics
        related_topics = data.get('RelatedTopics', [])
        for topic in related_topics[:num_results-1]:
            if isinstance(topic, dict) and topic.get('Text'):
                search_results.append({
                    'title': topic.get('Text', '')[:100] + '...',
                    'link': topic.get('FirstURL', ''),
                    'snippet': topic.get('Text', ''),
                    'displayLink': topic.get('FirstURL', '').split('/')[2] if topic.get('FirstURL') else 'duckduckgo.com',
                    'source': 'duckduckgo_related'
                })
        
        return search_results[:num_results]
    
    def convert_to_rag_context(self, web_results: List[Dict[str, Any]]) -> List[RAGContextItem]:
        """Convert free web search results to RAG context items"""
        rag_contexts = []
        
        for i, result in enumerate(web_results):
            try:
                # Create content from title and snippet
                content = f"{result.get('title', '')}. {result.get('snippet', '')}"
                
                rag_contexts.append(RAGContextItem(
                    chunk_id=i,
                    content=content[:2000],  # Limit content length
                    doc_type="free_web_search_result",
                    jurisdiction="general",
                    date=datetime.now().isoformat()[:10],
                    source_url=result.get('link', ''),
                    similarity=0.75  # Medium similarity for free web results
                ))
            except Exception as e:
                logger.warning(f"❌ Error converting free web result {i}: {e}")
                continue
        
        return rag_contexts


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
            model = getattr(settings, 'VERTEX_MODEL', 'gemini-2.0-flash')
            url = f"{self.base_url}/models/{model}:generateContent"
            
            headers = {
                'Content-Type': 'application/json',
                'X-Goog-Api-Key': self.api_key
            }
            
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
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.post(url, headers=headers, json=payload, timeout=30)
            )
            
            if response.status_code == 200:
                result = response.json()
                
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


class EnhancedRAGService:
    """Comprehensive RAG service with web search fallback"""
    
    def __init__(self):
        self.milvus_client = MilvusClient()
        self.vertex_client = VertexAIClient()
        self.web_search_service = VertexAISearchService()
        self.free_web_search_service = FreeWebSearchService()
        self.initialized = False
    
    async def initialize(self):
        """Initialize all RAG components"""
        if self.initialized:
            return
        
        logger.info("Initializing Enhanced RAG service...")
        
        await asyncio.gather(
            self.milvus_client.initialize(),
            self.vertex_client.initialize(),
            self.web_search_service.initialize(),
            self.free_web_search_service.initialize(),
            return_exceptions=True
        )
        
        self.initialized = True
        logger.info("Enhanced RAG service initialization complete")
    
    async def retrieve_contexts(self, query: str, top_k: int = 8, use_web_fallback: bool = True) -> List[RAGContextItem]:
        """Retrieve relevant contexts with strict fallback control"""
        if not self.initialized:
            await self.initialize()
        
        logger.info(f"🔍 Retrieving contexts for query: '{query[:50]}...'")
        
        
        
        try:
            # Step 1: Try Milvus RAG search first
            rag_contexts = await self._get_milvus_contexts(query, top_k)
            
            # Step 2: Evaluate quality of RAG results
            context_quality = self._evaluate_context_quality(rag_contexts, query)
            logger.info(f"📊 RAG context quality score: {context_quality:.2f}")
            
            # Step 3: Use web search fallback ONLY if explicitly enabled
            if context_quality < 0.5 and use_web_fallback:
                logger.info("🌐 RAG quality insufficient, using web search fallback")
                web_contexts = await self._get_web_contexts(query, min(top_k, 5))
                
                # Combine RAG and web results
                combined_contexts = rag_contexts + web_contexts
                combined_contexts.sort(key=lambda x: x.similarity, reverse=True)
                
                # Return best results from combined sources
                final_contexts = combined_contexts[:top_k]
                logger.info(f"✅ Combined {len(rag_contexts)} RAG + {len(web_contexts)} web contexts")
                return final_contexts
            elif not use_web_fallback:
                logger.info("🚫 Web fallback disabled, using RAG contexts only")
            
            logger.info(f"✅ Using {len(rag_contexts)} RAG contexts (quality sufficient)")
            return rag_contexts[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Context retrieval failed: {e}", exc_info=True)
            return []
    
    async def _get_milvus_contexts(self, query: str, top_k: int) -> List[RAGContextItem]:
        """Get contexts from Milvus vector database"""
        try:
            # Generate embeddings
            query_embeddings = await embed_texts_async([query])
            
            if not query_embeddings or not query_embeddings[0]:
                logger.error("❌ Failed to generate query embedding")
                return []
            
            query_vector = query_embeddings[0]
            logger.debug(f"🔢 Generated embedding vector of dimension: {len(query_vector)}")
            
            # Search Milvus
            milvus_results = await self.milvus_client.search(
                query_vector=query_vector,
                top_k=min(top_k, 8),
                metric_type="COSINE"
            )
            
            # Convert to RAG context items
            rag_contexts = []
            for i, result in enumerate(milvus_results):
                try:
                    similarity = float(result.get("similarity", 0.0))
                    
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
                        
                except Exception as e:
                    logger.warning(f"❌ Error processing result {i+1}: {e}")
                    continue
            
            rag_contexts.sort(key=lambda x: x.similarity, reverse=True)
            return rag_contexts
            
        except Exception as e:
            logger.error(f"❌ Milvus context retrieval failed: {e}")
            return []
    
    async def _get_web_contexts(self, query: str, num_results: int) -> List[RAGContextItem]:
        """Get contexts from web search with Vertex AI and free fallback"""
        try:
            # Enhance query for legal search
            legal_query = self._enhance_query_for_legal_search(query)
            
            web_contexts = []
            vertex_used = False
            
            # Try Vertex AI Search first
            try:
                vertex_results = await self.web_search_service.search_web(legal_query, num_results)
                if vertex_results:
                    vertex_contexts = self.web_search_service.convert_to_rag_context(vertex_results)
                    web_contexts.extend(vertex_contexts)
                    vertex_used = True
                    logger.info(f"🌐 Vertex AI search returned {len(vertex_contexts)} contexts")
            except Exception as e:
                logger.warning(f"Vertex AI search failed: {e}")
            
            # If Vertex search didn't return enough results, try free web search
            if len(web_contexts) < num_results // 2:
                try:
                    free_results = await self.free_web_search_service.search_web(legal_query, num_results - len(web_contexts))
                    if free_results:
                        free_contexts = self.free_web_search_service.convert_to_rag_context(free_results)
                        web_contexts.extend(free_contexts)
                        logger.info(f"🌐 Free web search returned {len(free_contexts)} contexts")
                except Exception as e:
                    logger.warning(f"Free web search failed: {e}")
            
            # Log which path was taken
            logger.info(f"vertex_used={vertex_used}")
            logger.info(f"🌐 Total web search returned {len(web_contexts)} contexts")
            return web_contexts
            
        except Exception as e:
            logger.error(f"❌ Web context retrieval failed: {e}")
            return []
    
    def _enhance_query_for_legal_search(self, query: str) -> str:
        """Enhance query with legal context for better web search results"""
        # Add legal context terms if not already present
        legal_terms = ["law", "legal", "contract", "clause", "agreement", "terms"]
        query_lower = query.lower()
        
        if not any(term in query_lower for term in legal_terms):
            # Add legal context to make search more relevant
            if "?" in query:
                enhanced_query = f"{query} law legal"
            else:
                enhanced_query = f"{query} legal law contract"
        else:
            enhanced_query = query
        
        logger.debug(f"Enhanced query: '{enhanced_query}'")
        return enhanced_query
    
    def _evaluate_context_quality(self, contexts: List[RAGContextItem], query: str) -> float:
        """Evaluate the quality of retrieved contexts"""
        if not contexts:
            return 0.0
        
        # Quality metrics
        avg_similarity = sum(ctx.similarity for ctx in contexts) / len(contexts)
        high_quality_count = sum(1 for ctx in contexts if ctx.similarity > 0.7)
        content_relevance = self._assess_content_relevance(contexts, query)
        
        # Combined quality score (0.0 to 1.0)
        quality_score = (
            avg_similarity * 0.4 +
            (high_quality_count / len(contexts)) * 0.3 +
            content_relevance * 0.3
        )
        
        return min(quality_score, 1.0)
    
    def _assess_content_relevance(self, contexts: List[RAGContextItem], query: str) -> float:
        """Assess content relevance based on keyword overlap"""
        if not contexts:
            return 0.0
        
        query_words = set(query.lower().split())
        relevant_count = 0
        
        for context in contexts:
            content_words = set(context.content.lower().split())
            overlap = len(query_words & content_words)
            if overlap >= max(1, len(query_words) * 0.2):  # At least 20% overlap
                relevant_count += 1
        
        return relevant_count / len(contexts)
    
    async def answer_with_vertex(self, question: str, contexts: List[RAGContextItem], summary_hint: Optional[str] = None) -> str:
        """Generate an answer using Vertex AI with enhanced context handling"""
        
        context_count = len(contexts)
        
        # Separate RAG and web contexts for better prompting
        rag_contexts = [ctx for ctx in contexts if ctx.doc_type != "web_search_result"]
        web_contexts = [ctx for ctx in contexts if ctx.doc_type == "web_search_result"]
        
        # Build enhanced prompt
        context_text = ""
        evidence_topics = []
        
        if rag_contexts:
            rag_text = "\n\n".join([f"Legal Document {i+1}:\n{ctx.content}" for i, ctx in enumerate(rag_contexts[:3], 1)])
            context_text += f"LEGAL DATABASE EVIDENCE:\n{rag_text}\n\n"
            
            # Extract topics from legal database
            for ctx in rag_contexts:
                if "liability" in ctx.content.lower():
                    evidence_topics.append("liability law")
                if "contract" in ctx.content.lower():
                    evidence_topics.append("contract law")
                if "risk" in ctx.content.lower():
                    evidence_topics.append("legal risk")
        
        if web_contexts:
            web_text = "\n\n".join([f"Web Source {i+1}:\n{ctx.content}" for i, ctx in enumerate(web_contexts[:2], 1)])
            context_text += f"CURRENT WEB INFORMATION:\n{web_text}\n\n"
        
        summary_context = f"Document Summary: {summary_hint}\n\n" if summary_hint else ""
        
        # Enhanced prompt that handles both legal and general questions with Indian law focus
        prompt = f"""You are a knowledgeable AI assistant with expertise in Indian law and general knowledge. You specialize in providing clear, accurate answers based on Indian legal framework and current information.

QUESTION: {question}

AVAILABLE EVIDENCE:
- Found {len(rag_contexts)} legal database documents
- Found {len(web_contexts)} current web sources
- Topics covered: {', '.join(evidence_topics) if evidence_topics else 'various topics'}

{context_text}
{summary_context}

INSTRUCTIONS:
1. For legal questions: Prioritize Indian law context and legal database evidence. Provide comprehensive analysis based on Indian legal framework, citing relevant laws, acts, and precedents where applicable.
2. For general questions: Use all available sources to provide accurate, current information with Indian context when relevant.
3. Always provide helpful, substantive answers that are practical and actionable.
4. Use clear, accessible language that non-lawyers can understand.
5. If combining legal and general information, clearly distinguish between them.
6. Include practical recommendations and next steps where appropriate.
7. When discussing legal matters, always mention that this is general information and recommend consulting with a qualified Indian lawyer for specific legal advice.
8. For Indian law questions, reference relevant Indian legal acts, sections, and case laws when available.

Provide a thorough, well-structured response with Indian legal context:"""
        
        try:
            answer = await self.vertex_client.generate_text(
                prompt=prompt,
                max_tokens=getattr(settings, 'VERTEX_MAX_TOKENS', 1024),
                temperature=getattr(settings, 'VERTEX_TEMPERATURE', 0.3)
            )
            
            # Retry logic for inadequate responses
            if self._is_inadequate_response(answer):
                logger.info("Response seems inadequate, retrying with enhanced prompt")
                enhanced_answer = await self._retry_with_enhanced_prompt(question, contexts, summary_hint)
                return enhanced_answer or answer
            
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"I found {context_count} relevant sources but encountered an error generating the response. Please try rephrasing your question."

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
        
        context_text = "\n\n".join([f"Source {i+1}: {ctx.content}" for i, ctx in enumerate(contexts[:3])])
        
        enhanced_prompt = f"""You are a knowledgeable assistant. The user asked a question and I've found relevant information sources.

TASK: Provide a comprehensive answer using the available information and your knowledge.

QUESTION: {question}

AVAILABLE INFORMATION:
{context_text}

REQUIREMENTS:
- Answer the question directly and thoroughly
- Use the provided information as supporting evidence
- Apply relevant knowledge even if sources don't directly address all aspects
- Make the response helpful and informative
- Use clear, accessible language

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
    
    async def summarize_500w_with_recommendations(self, text: str) -> str:
        """Generate a comprehensive summary with recommendations in ≤500 words"""
        if not text or not text.strip():
            logger.warning("No text provided for summary generation")
            return "No content available for summary."
        
        logger.info(f"Generating enhanced summary for {len(text)} characters")
        
        prompt = f"""You are a legal document assistant specializing in Indian law. Create a comprehensive summary of this legal document in exactly 500 words or fewer, followed by practical recommendations.

SUMMARY SECTION (400 words max):
Focus on:
1. Main purpose and type of document
2. Key parties involved and their roles
3. Primary obligations and rights of each party
4. Important terms and conditions
5. Notable risks or considerations
6. Governing law and jurisdiction
7. Key financial terms and payment obligations
8. Termination and dispute resolution clauses

RECOMMENDATIONS SECTION (100 words max):
Provide practical recommendations:
1. Key areas requiring legal review
2. Potential risks to watch out for
3. Suggested next steps
4. Important clauses to negotiate or clarify

Use simple, non-technical language and focus on Indian legal context.

Document text:
{text[:6000]}...

Comprehensive Summary with Recommendations (≤500 words):"""
        
        try:
            summary = await self.vertex_client.generate_text(
                prompt=prompt,
                max_tokens=600,  # Increased for longer summary
                temperature=0.1
            )
            
            if not summary or not summary.strip():
                logger.warning("Vertex AI returned empty summary")
                return self._create_fallback_summary_with_recommendations(text)
            
            # Ensure word limit
            words = summary.split()
            if len(words) > 500:
                summary = " ".join(words[:500]) + "..."
            
            logger.info(f"Generated enhanced summary: {len(summary)} chars, {len(words)} words")
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}", exc_info=True)
            return self._create_fallback_summary_with_recommendations(text)
    
    def _create_fallback_summary(self, text: str) -> str:
        """Create a simple fallback summary"""
        words = text.split()
        if len(words) <= 150:
            return text
        
        sample_text = " ".join(words[:150])
        return f"This legal document contains {len(words)} words covering contractual terms and conditions. {sample_text}..."
    
    def _create_fallback_summary_with_recommendations(self, text: str) -> str:
        """Create a fallback summary with recommendations"""
        words = text.split()
        word_count = len(words)
        
        if word_count <= 200:
            return f"This legal document contains {word_count} words covering various contractual terms and conditions. {text}\n\nRECOMMENDATIONS: Please review this document carefully and consider consulting with a qualified Indian lawyer for specific legal advice regarding your situation."
        
        sample_text = " ".join(words[:200])
        return f"""SUMMARY: This legal document contains {word_count} words covering various contractual terms and conditions. The document appears to establish rights, obligations, and procedures for the parties involved. Key areas typically include liability, termination, payment terms, and governing law. {sample_text}...

RECOMMENDATIONS: 
1. Review all liability and indemnification clauses carefully
2. Check payment terms and deadlines
3. Understand termination conditions and notice requirements
4. Verify governing law and jurisdiction clauses
5. Consider consulting with a qualified Indian lawyer for specific legal advice
6. Ensure all parties understand their rights and obligations"""
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        return {
            "initialized": self.initialized,
            "milvus_connected": self.milvus_client.connected,
            "vertex_initialized": self.vertex_client.initialized,
            "web_search_initialized": self.web_search_service.initialized,
            "free_web_search_initialized": self.free_web_search_service.initialized,
            "collection_name": getattr(settings, 'MILVUS_COLLECTION', 'unknown'),
            "embedding_dimension": 768,
            "dependencies": {
                "pymilvus_available": PYMILVUS_AVAILABLE,
                "google_search_available": GOOGLE_SEARCH_AVAILABLE,
                "vertex_search_available": VERTEX_SEARCH_AVAILABLE,
                "async_http_available": ASYNC_HTTP_AVAILABLE,
                "requests_available": True,
                "http_client": True
            }
        }


# Global service instance
_rag_service = None


async def get_rag_service() -> EnhancedRAGService:
    """Get or create the global RAG service instance"""
    global _rag_service
    if _rag_service is None:
        _rag_service = EnhancedRAGService()
        await _rag_service.initialize()
    return _rag_service


# Public API functions for router integration
async def retrieve_contexts(query: str, top_k: int = 8, use_web_fallback: bool = True) -> List[RAGContextItem]:
    """Retrieve relevant contexts for a query with web search fallback"""
    service = await get_rag_service()
    return await service.retrieve_contexts(query, top_k, use_web_fallback)


async def summarize_200w(text: str) -> str:
    """Generate a summary in ≤200 words"""
    service = await get_rag_service()
    return await service.summarize_200w(text)


async def summarize_500w_with_recommendations(text: str) -> str:
    """Generate a comprehensive summary with recommendations in ≤500 words"""
    service = await get_rag_service()
    return await service.summarize_500w_with_recommendations(text)


async def answer_with_vertex(question: str, contexts: List[RAGContextItem], summary_hint: Optional[str] = None) -> str:
    """Generate an answer using Vertex AI with RAG contexts and web search fallback"""
    service = await get_rag_service()
    return await service.answer_with_vertex(question, contexts, summary_hint)


async def health_check() -> Dict[str, Any]:
    """Health check for enhanced RAG service"""
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