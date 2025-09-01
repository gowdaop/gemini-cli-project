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
            "document_summary": None
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
        evidence: List[RAGContextItem]
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
            "evidence_count": len(evidence)
        })
        
        # Trim history if too long
        if len(conversation["history"]) > MAX_CONVERSATION_HISTORY:
            conversation["history"] = conversation["history"][-MAX_CONVERSATION_HISTORY:]
        
        conversation["last_activity"] = datetime.utcnow()
        logger.debug(f"Updated conversation {conversation_id} with new Q&A")
    
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


class QuestionProcessor:
    """Processes and validates user questions"""
    
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
    def extract_question_intent(question: str) -> Dict[str, Any]:
        """Extract intent and keywords from question"""
        question_lower = question.lower()
        
        # Detect question type
        intent = {
            "type": "general",
            "keywords": [],
            "is_definition": False,
            "is_risk_related": False,
            "is_clause_specific": False,
            "urgency": "normal"
        }
        
        # Definition questions
        if any(phrase in question_lower for phrase in [
            "what is", "what does", "define", "meaning of", "explain"
        ]):
            intent["type"] = "definition"
            intent["is_definition"] = True
        
        # Risk-related questions
        if any(phrase in question_lower for phrase in [
            "risk", "danger", "liability", "problem", "issue", "concern", "safe"
        ]):
            intent["is_risk_related"] = True
        
        # Clause-specific questions
        if any(phrase in question_lower for phrase in [
            "clause", "section", "paragraph", "term", "condition"
        ]):
            intent["is_clause_specific"] = True
        
        # Urgency detection
        if any(phrase in question_lower for phrase in [
            "urgent", "immediately", "asap", "emergency", "critical"
        ]):
            intent["urgency"] = "high"
        
        # Extract legal keywords
        legal_keywords = [
            "liability", "indemnification", "termination", "breach", "damages",
            "confidentiality", "intellectual property", "payment", "arbitration",
            "governing law", "force majeure", "assignment", "modification"
        ]
        
        intent["keywords"] = [
            keyword for keyword in legal_keywords 
            if keyword in question_lower
        ]
        
        return intent


class ChatService:
    """Main chat service that integrates RAG and conversation management"""
    
    def __init__(self):
        self.rag_service = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize the RAG service"""
        if not self.initialized:
            self.rag_service = await rag.get_rag_service()
            self.initialized = True
            logger.info("Chat service initialized with RAG")
    
    async def handle_chat_request(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        ocr: Optional[OCRText] = None,
        summary_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle a complete chat request with RAG integration"""
        
        # Initialize if needed
        if not self.initialized:
            await self.initialize()
        
        # Validate and process question
        question = QuestionProcessor.validate_question(question)
        question_intent = QuestionProcessor.extract_question_intent(question)
        
        logger.info(f"Processing chat question: {question[:100]}...")
        logger.debug(f"Question intent: {question_intent}")
        
        # Get or create conversation
        if conversation_id:
            conversation = ConversationManager.get_conversation(conversation_id)
            if not conversation:
                logger.warning(f"Conversation {conversation_id} not found, creating new one")
                conversation_id = ConversationManager.create_conversation()
                conversation = ConversationManager.get_conversation(conversation_id)
        else:
            conversation_id = ConversationManager.create_conversation()
            conversation = ConversationManager.get_conversation(conversation_id)
        
        # Store document context if provided
        if ocr and conversation:
            conversation["context"]["current_document"] = ocr.dict()
            if summary_hint:
                conversation["document_summary"] = summary_hint
        
        # Retrieve relevant contexts using RAG
        logger.debug("Retrieving legal contexts from RAG...")
        evidence = await self._retrieve_comprehensive_evidence(
            question, 
            question_intent,
            ocr,
            conversation
        )
        
        logger.info(f"Retrieved {len(evidence)} evidence items")
        
        # Generate intelligent response using RAG + Vertex AI
        logger.debug("Generating AI response...")
        answer = await rag.answer_with_vertex(
            question=question,
            contexts=evidence,
            summary_hint=conversation.get("document_summary")
        )
        
        # Update conversation history
        ConversationManager.update_conversation(
            conversation_id, question, answer, evidence
        )
        
        logger.info(f"Chat response generated successfully for conversation {conversation_id}")
        
        return {
            "answer": answer,
            "evidence": evidence,
            "conversation_id": conversation_id
        }
    
    async def _retrieve_comprehensive_evidence(
        self,
        question: str,
        question_intent: Dict[str, Any],
        document_context: Optional[OCRText] = None,
        conversation: Optional[Dict] = None
    ) -> List[RAGContextItem]:
        """Retrieve evidence from multiple sources"""
        try:
            evidence_items = []
            
            # Primary: Retrieve from Milvus using RAG service
            logger.debug("Searching Milvus for relevant legal precedents...")
            milvus_evidence = await rag.retrieve_contexts(question, top_k=5)
            evidence_items.extend(milvus_evidence)
            
            # Secondary: Search within current document if provided
            if document_context and document_context.blocks:
                logger.debug("Searching current document context...")
                document_evidence = self._search_document_context(question, document_context)
                evidence_items.extend(document_evidence)
            
            # Sort by relevance/similarity score
            evidence_items.sort(key=lambda x: x.similarity, reverse=True)
            
            # Return top 8 items to avoid overwhelming the LLM
            final_evidence = evidence_items[:8]
            logger.info(f"Compiled {len(final_evidence)} total evidence items")
            
            return final_evidence
            
        except Exception as e:
            logger.error(f"Evidence retrieval failed: {e}")
            return []
    
    def _search_document_context(self, question: str, document: OCRText) -> List[RAGContextItem]:
        """Search within the current document for relevant sections"""
        relevant_blocks = []
        question_lower = question.lower()
        question_words = [word for word in question_lower.split() if len(word) > 3]
        
        for i, block in enumerate(document.blocks):
            block_lower = block.text.lower()
            
            # Enhanced keyword matching
            relevance_score = 0
            for word in question_words:
                if word in block_lower:
                    relevance_score += 1
            
            # Only include blocks with reasonable relevance
            if relevance_score > 0:
                similarity = relevance_score / len(question_words) if question_words else 0
                
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


# Initialize global chat service
_chat_service = ChatService()


@router.post("/", response_model=ChatResponse)
async def chat_with_document(
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    """
    Chat with legal document using RAG and LLM
    
    Provides intelligent Q&A about legal documents with:
    - Context-aware responses using Vertex AI
    - Legal precedent citations from Milvus
    - Simplified explanations for non-lawyers
    - Conversation continuity
    """
    try:
        # Handle the chat request using our service
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
    """Get conversation history"""
    conversation = ConversationManager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or expired"
        )
    
    return {
        "conversation_id": conversation_id,
        "created_at": conversation["created_at"].isoformat(),
        "last_activity": conversation["last_activity"].isoformat(),
        "history": conversation["history"],
        "message_count": len(conversation["history"])
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


@router.get("/health")
async def chat_health():
    """Health check for chat service"""
    active_conversations = len(conversation_store)
    return {
        "status": "healthy",
        "service": "chat",
        "active_conversations": active_conversations,
        "features": [
            "rag_integration",
            "vertex_ai_responses", 
            "conversation_management",
            "legal_simplification",
            "document_context_search"
        ]
    }


@router.get("/stats")
async def chat_statistics():
    """Get chat service statistics"""
    total_conversations = len(conversation_store)
    total_messages = sum(
        len(conv["history"]) for conv in conversation_store.values()
    )
    
    return {
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "average_messages_per_conversation": (
            total_messages / total_conversations if total_conversations > 0 else 0
        ),
        "conversation_timeout_hours": CONVERSATION_TIMEOUT.total_seconds() / 3600,
        "rag_integration": "enabled",
        "vertex_ai_integration": "enabled"
    }
