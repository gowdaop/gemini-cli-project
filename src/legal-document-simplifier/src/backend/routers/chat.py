from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Optional, List, Dict, Any
import logging
import asyncio
import json
import uuid
from datetime import datetime, timedelta
import hashlib
from contextlib import asynccontextmanager
import numpy as np
from ..schemas.analysis import (
    ChatRequest, 
    ChatResponse, 
    RAGContextItem, 
    ErrorResponse,
    OCRText
)
from ..services import rag, embedding
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        404: {"model": ErrorResponse, "description": "Conversation Not Found"},
        429: {"model": ErrorResponse, "description": "Rate Limit Exceeded"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    }
)

# In-memory conversation store (will be replaced with PostgreSQL later)
conversation_store: Dict[str, Dict[str, Any]] = {}
CONVERSATION_TIMEOUT = timedelta(hours=2)
MAX_CONVERSATION_HISTORY = 10
MAX_QUESTION_LENGTH = 1000


class ConversationManager:
    """Manages conversation state and history"""
    
    @staticmethod
    def create_conversation(user_context: Optional[Dict] = None) -> str:
        """Create a new conversation with unique ID"""
        conversation_id = str(uuid.uuid4())
        conversation_store[conversation_id] = {
            "id": conversation_id,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "history": [],
            "context": user_context or {},
            "document_summary": None,
            "question_type_history": []  # Track question types for better routing
        }
        logger.info(f"Created new conversation: {conversation_id}")
        return conversation_id
    
    @staticmethod
    def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve conversation by ID"""
        conversation = conversation_store.get(conversation_id)
        if not conversation:
            return None
            
        # Check if conversation has expired
        if datetime.utcnow() - conversation["last_activity"] > CONVERSATION_TIMEOUT:
            ConversationManager.cleanup_conversation(conversation_id)
            return None
            
        return conversation
    
    @staticmethod
    def update_conversation(
        conversation_id: str, 
        question: str, 
        answer: str, 
        evidence: List[RAGContextItem],
        question_type: str = "general"
    ) -> None:
        """Update conversation with new Q&A pair"""
        conversation = conversation_store.get(conversation_id)
        if not conversation:
            return
            
        # Add to history
        conversation["history"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "question": question,
            "answer": answer,
            "evidence_count": len(evidence),
            "question_type": question_type,
            "sources_used": [ctx.doc_type for ctx in evidence]
        })
        
        # Track question types for better routing
        conversation["question_type_history"].append(question_type)
        if len(conversation["question_type_history"]) > 5:
            conversation["question_type_history"] = conversation["question_type_history"][-5:]
        
        # Trim history if too long
        if len(conversation["history"]) > MAX_CONVERSATION_HISTORY:
            conversation["history"] = conversation["history"][-MAX_CONVERSATION_HISTORY:]
        
        conversation["last_activity"] = datetime.utcnow()
        logger.debug(f"Updated conversation {conversation_id} with new Q&A (type: {question_type})")
    
    @staticmethod
    def cleanup_conversation(conversation_id: str) -> None:
        """Remove expired conversation"""
        if conversation_id in conversation_store:
            del conversation_store[conversation_id]
            logger.info(f"Cleaned up expired conversation: {conversation_id}")
    
    @staticmethod
    def cleanup_expired_conversations() -> None:
        """Clean up all expired conversations"""
        now = datetime.utcnow()
        expired_ids = [
            conv_id for conv_id, conv in conversation_store.items()
            if now - conv["last_activity"] > CONVERSATION_TIMEOUT
        ]
        
        for conv_id in expired_ids:
            ConversationManager.cleanup_conversation(conv_id)
        
        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired conversations")


class EnhancedQuestionProcessor:
    """Enhanced question processing with better intent detection"""
    
    @staticmethod
    def validate_question(question: str) -> str:
        """Validate and clean user question"""
        if not question or not question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )
        
        question = question.strip()
        
        if len(question) > MAX_QUESTION_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Question too long. Maximum {MAX_QUESTION_LENGTH} characters."
            )
        
        return question
    
    @staticmethod
    def extract_question_intent(question: str, conversation_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Enhanced intent extraction with context awareness"""
        question_lower = question.lower()
        
        intent = {
            "type": "general",
            "keywords": [],
            "is_legal": False,
            "is_document_specific": False,
            "requires_web_search": False,
            "urgency": "normal",
            "confidence": 0.5
        }
        
        # Legal question detection (enhanced patterns)
        legal_indicators = [
            "contract", "clause", "liability", "legal", "law", "agreement", "terms",
            "indemnification", "breach", "termination", "payment terms", "governing law",
            "arbitration", "confidentiality", "intellectual property", "damages",
            "jurisdiction", "force majeure", "warranties", "representations"
        ]
        
        legal_score = sum(1 for indicator in legal_indicators if indicator in question_lower)
        if legal_score >= 2:
            intent["is_legal"] = True
            intent["type"] = "legal"
            intent["confidence"] = min(0.9, 0.5 + (legal_score * 0.1))
        
        # Document-specific question detection
        document_indicators = [
            "this document", "this contract", "this agreement", "in the document",
            "what does this", "explain this", "this clause", "this section"
        ]
        
        if any(indicator in question_lower for indicator in document_indicators):
            intent["is_document_specific"] = True
            intent["confidence"] = max(intent["confidence"], 0.8)
        
        # Web search requirement detection
        web_search_indicators = [
            "current", "latest", "recent", "today", "now", "2024", "2025",
            "news", "update", "what's happening", "current status", "recent changes",
            "new law", "recent case", "current regulation", "market", "stock",
            "weather", "sports", "technology news", "breaking news"
        ]
        
        if any(indicator in question_lower for indicator in web_search_indicators):
            intent["requires_web_search"] = True
            intent["type"] = "current_events" if not intent["is_legal"] else "current_legal"
            intent["confidence"] = max(intent["confidence"], 0.7)
        
        # General knowledge questions that might need web search
        general_web_indicators = [
            "how to", "what is", "who is", "when did", "where is", "why does",
            "explain", "definition", "meaning", "compare", "difference between"
        ]
        
        general_score = sum(1 for indicator in general_web_indicators if indicator in question_lower)
        if general_score >= 1 and not intent["is_legal"]:
            intent["type"] = "general_knowledge"
            intent["requires_web_search"] = True
            intent["confidence"] = max(intent["confidence"], 0.6)
        
        # Context-aware adjustments
        if conversation_context:
            recent_types = conversation_context.get("question_type_history", [])
            if recent_types and "legal" in recent_types[-2:]:
                # If recent questions were legal, bias towards legal interpretation
                if intent["confidence"] < 0.7:
                    intent["confidence"] += 0.1
        
        # Extract keywords
        all_indicators = legal_indicators + document_indicators + web_search_indicators + general_web_indicators
        intent["keywords"] = [keyword for keyword in all_indicators if keyword in question_lower]
        
        return intent


class EnhancedChatService:
    """Enhanced chat service with intelligent routing"""
    
    def __init__(self):
        self.rag_service = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize the RAG service"""
        if not self.initialized:
            self.rag_service = await rag.get_rag_service()
            self.initialized = True
            logger.info("Enhanced chat service initialized with RAG and web search")
    
    async def handle_chat_request(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        ocr: Optional[OCRText] = None,
        summary_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """Enhanced chat request handling with intelligent routing"""
        
        # Initialize if needed
        if not self.initialized:
            await self.initialize()
        
        # Validate and process question
        question = EnhancedQuestionProcessor.validate_question(question)
        
        # Get or create conversation
        if conversation_id:
            conversation = ConversationManager.get_conversation(conversation_id)
            if not conversation:
                logger.warning(f"Conversation {conversation_id} not found or expired, creating a new one.")
                conversation_id = ConversationManager.create_conversation()
        else:
            conversation_id = ConversationManager.create_conversation()
        
        conversation = ConversationManager.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create or retrieve conversation"
            )
        
        # Enhanced intent detection with conversation context
        question_intent = EnhancedQuestionProcessor.extract_question_intent(
            question, conversation
        )
        
        logger.info(f"Processing question (type: {question_intent['type']}, confidence: {question_intent['confidence']:.2f})")
        logger.debug(f"Question intent: {question_intent}")
        
        # Store document context if provided
        if ocr and conversation:
            conversation["context"]["current_document"] = ocr.dict()
            if summary_hint:
                conversation["document_summary"] = summary_hint
        
        # Intelligent evidence retrieval based on question type
        logger.debug("Retrieving contexts with intelligent routing...")
        evidence = await self._retrieve_intelligent_evidence(
            question, 
            question_intent,
            ocr,
            conversation
        )
        
        logger.info(f"Retrieved {len(evidence)} evidence items")
        
        # Generate response using enhanced RAG
        logger.debug("Generating AI response...")
        answer = await rag.answer_with_vertex(
            question=question,
            contexts=evidence,
            summary_hint=conversation.get("document_summary")
        )
        
        # Update conversation history with enhanced tracking
        ConversationManager.update_conversation(
            conversation_id, question, answer, evidence, question_intent["type"]
        )
        
        logger.info(f"Chat response generated successfully for conversation {conversation_id}")
        
        return {
            "answer": answer,
            "evidence": evidence,
            "conversation_id": conversation_id,
            "question_type": question_intent["type"],
            "sources_used": self._get_source_summary(evidence)
        }
    
    async def _retrieve_intelligent_evidence(
        self,
        question: str,
        question_intent: Dict[str, Any],
        document_context: Optional[OCRText] = None,
        conversation: Optional[Dict] = None
    ) -> List[RAGContextItem]:
        """Intelligent evidence retrieval based on question intent"""
        try:
            evidence_items = []
            
            # Strategy 1: Document-specific questions - prioritize current document
            if question_intent["is_document_specific"] and document_context:
                logger.debug("Prioritizing current document context...")
                document_evidence = self._search_document_context(question, document_context)
                evidence_items.extend(document_evidence)
                
                # Also get some legal precedents for context
                if question_intent["is_legal"]:
                    legal_evidence = await rag.retrieve_contexts(question, top_k=3, use_web_fallback=False)
                    evidence_items.extend(legal_evidence)
            
            # Strategy 2: Legal questions - use legal database first, web search as fallback
            elif question_intent["is_legal"]:
                logger.debug("Using legal database with web search fallback...")
                legal_evidence = await rag.retrieve_contexts(
                    question, 
                    top_k=6, 
                    use_web_fallback=True  # Enable web search fallback for legal questions
                )
                evidence_items.extend(legal_evidence)
                
                # Add current document context if available
                if document_context:
                    document_evidence = self._search_document_context(question, document_context)
                    evidence_items.extend(document_evidence[:2])  # Limit document context
            
            # Strategy 3: Current events or general knowledge - prioritize web search
            elif question_intent["requires_web_search"] or question_intent["type"] in ["current_events", "general_knowledge"]:
                logger.debug("Prioritizing web search for current/general information...")
                web_evidence = await rag.retrieve_contexts(
                    question, 
                    top_k=8, 
                    use_web_fallback=True  # Enable web search for general questions
                )
                evidence_items.extend(web_evidence)
            
            # Strategy 4: Generic questions - balanced approach
            else:
                logger.debug("Using balanced evidence retrieval...")
                balanced_evidence = await rag.retrieve_contexts(question, top_k=6, use_web_fallback=True)
                evidence_items.extend(balanced_evidence)
                
                # Add document context if available
                if document_context:
                    document_evidence = self._search_document_context(question, document_context)
                    evidence_items.extend(document_evidence[:2])
            
            # Remove duplicates and sort by relevance
            seen_content = set()
            unique_evidence = []
            for item in evidence_items:
                content_hash = hashlib.md5(item.content[:200].encode()).hexdigest()
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_evidence.append(item)
            
            # Sort by similarity score
            unique_evidence.sort(key=lambda x: x.similarity, reverse=True)
            
            # Return top items
            final_evidence = unique_evidence[:8]
            logger.info(f"Compiled {len(final_evidence)} unique evidence items")
            
            return final_evidence
            
        except Exception as e:
            logger.error(f"Intelligent evidence retrieval failed: {e}")
            # Fallback to basic retrieval
            try:
                return await rag.retrieve_contexts(question, top_k=5, use_web_fallback=True)
            except:
                return []
    
    def _search_document_context(self, question: str, document: OCRText) -> List[RAGContextItem]:
        """Enhanced document context search"""
        relevant_blocks = []
        question_lower = question.lower()
        question_words = [word for word in question_lower.split()]
        
        for i, block in enumerate(document.blocks):
            block_lower = block.text.lower()
            
            # Enhanced keyword matching with phrase detection
            relevance_score = 0
            
            # Exact phrase matching (higher weight)
            question_phrases = [' '.join(question_words[i:i+2]) for i in range(len(question_words)-1)]
            for phrase in question_phrases:
                if phrase in block_lower:
                    relevance_score += 2
            
            # Individual word matching
            for word in question_words:
                if word in block_lower:
                    relevance_score += 1
            
            # Context relevance (legal terms, document structure)
            context_indicators = ["clause", "section", "agreement", "shall", "party", "terms"]
            for indicator in context_indicators:
                if indicator in block_lower:
                    relevance_score += 0.5
            
            # Only include blocks with reasonable relevance
            if relevance_score > 1:
                similarity = min(0.9, relevance_score / (len(question_words) + 2))
                
                relevant_blocks.append(RAGContextItem(
                    chunk_id=i,
                    content=block.text,
                    doc_type="current_document",
                    jurisdiction="current",
                    date=datetime.now().isoformat(),
                    source_url="current_document",
                    similarity=similarity
                ))
        
        # Return top 3 most relevant blocks from current document
        return sorted(relevant_blocks, key=lambda x: x.similarity, reverse=True)[:3]
    
    def _get_source_summary(self, evidence: List[RAGContextItem]) -> Dict[str, int]:
        """Get summary of sources used"""
        sources = {}
        for item in evidence:
            source_type = item.doc_type
            sources[source_type] = sources.get(source_type, 0) + 1
        return sources


# Initialize global chat service
_chat_service = EnhancedChatService()


@router.post("/", response_model=ChatResponse)
async def chat_with_document(
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    """
    Enhanced chat with legal document using RAG and web search
    
    Provides intelligent Q&A with:
    - Context-aware responses using Vertex AI
    - Legal precedent citations from Milvus
    - Web search fallback for current information
    - Document-specific analysis
    - Simplified explanations for non-lawyers
    - Conversation continuity
    """
    try:
        # Handle the chat request using enhanced service
        response_data = await _chat_service.handle_chat_request(
            question=request.question,
            conversation_id=request.conversation_id,
            ocr=request.ocr,
            summary_hint=request.summary_hint
        )
        
        # Schedule cleanup in background
        background_tasks.add_task(ConversationManager.cleanup_expired_conversations)
        
        return ChatResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing failed: {str(e)}"
        )


@router.get("/conversations/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """Get enhanced conversation history with analytics"""
    conversation = ConversationManager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or expired"
        )
    
    # Enhanced analytics
    question_types = [msg.get("question_type", "unknown") for msg in conversation["history"]]
    source_usage = {}
    for msg in conversation["history"]:
        for source_type in msg.get("sources_used", []):
            source_usage[source_type] = source_usage.get(source_type, 0) + 1
    
    return {
        "conversation_id": conversation_id,
        "created_at": conversation["created_at"].isoformat(),
        "last_activity": conversation["last_activity"].isoformat(),
        "history": conversation["history"],
        "message_count": len(conversation["history"]),
        "analytics": {
            "question_types": dict(zip(*np.unique(question_types, return_counts=True))) if question_types else {},
            "source_usage": source_usage,
            "has_document_context": "current_document" in conversation.get("context", {})
        }
    }


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    if conversation_id not in conversation_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    ConversationManager.cleanup_conversation(conversation_id)
    return {"message": "Conversation deleted successfully"}


@router.post("/new", response_model=Dict[str, str])
async def new_chat():
    """Create a new, empty chat conversation"""
    try:
        conversation_id = ConversationManager.create_conversation()
        logger.info(f"New enhanced conversation created: {conversation_id}")
        return {"conversation_id": conversation_id}
    except Exception as e:
        logger.error(f"Failed to create new conversation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create new conversation"
        )


@router.get("/health")
async def chat_health():
    """Enhanced health check for chat service"""
    active_conversations = len(conversation_store)
    
    # Get service status
    rag_stats = await rag.health_check()
    
    return {
        "status": "healthy",
        "service": "enhanced_chat",
        "active_conversations": active_conversations,
        "rag_service_status": rag_stats.get("status", "unknown"),
        "features": [
            "intelligent_question_routing",
            "web_search_integration",
            "legal_database_integration",
            "document_context_analysis",
            "conversation_analytics",
            "multi_source_evidence"
        ]
    }


@router.get("/stats")
async def chat_statistics():
    """Enhanced chat service statistics"""
    total_conversations = len(conversation_store)
    total_messages = sum(
        len(conv["history"]) for conv in conversation_store.values()
    )
    
    # Question type analysis
    all_question_types = []
    source_usage_totals = {}
    
    for conv in conversation_store.values():
        for msg in conv["history"]:
            all_question_types.append(msg.get("question_type", "unknown"))
            for source in msg.get("sources_used", []):
                source_usage_totals[source] = source_usage_totals.get(source, 0) + 1
    
    return {
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "average_messages_per_conversation": (
            total_messages / total_conversations if total_conversations > 0 else 0
        ),
        "conversation_timeout_hours": CONVERSATION_TIMEOUT.total_seconds() / 3600,
        "question_type_distribution": dict(zip(*np.unique(all_question_types, return_counts=True))) if all_question_types else {},
        "source_usage_totals": source_usage_totals,
        "integrations": {
            "rag_integration": "enabled",
            "web_search_integration": "enabled", 
            "vertex_ai_integration": "enabled",
            "document_analysis": "enabled"
        }
    }