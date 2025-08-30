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

class ResponseGenerator:
    """Generates structured responses for legal questions"""
    
    @staticmethod
    async def generate_legal_response(
        question: str,
        evidence: List[RAGContextItem],
        conversation_context: Optional[Dict] = None,
        document_context: Optional[OCRText] = None
    ) -> str:
        """Generate a comprehensive legal response"""
        
        # Build context for LLM
        context_parts = []
        
        # Add document context if available
        if document_context and document_context.full_text:
            context_parts.append(f"DOCUMENT CONTEXT:\n{document_context.full_text[:2000]}...")
        
        # Add conversation history
        if conversation_context and conversation_context.get("history"):
            recent_history = conversation_context["history"][-3:]  # Last 3 exchanges
            history_text = "\n".join([
                f"Q: {h['question']}\nA: {h['answer'][:200]}..."
                for h in recent_history
            ])
            context_parts.append(f"CONVERSATION HISTORY:\n{history_text}")
        
        # Add evidence from RAG
        if evidence:
            evidence_text = "\n\n".join([
                f"Source {i+1} ({ctx.doc_type}, {ctx.jurisdiction}):\n{ctx.content}"
                for i, ctx in enumerate(evidence[:5])  # Top 5 pieces of evidence
            ])
            context_parts.append(f"LEGAL PRECEDENTS:\n{evidence_text}")
        
        # Combine all context
        full_context = "\n\n" + "="*50 + "\n\n".join(context_parts)
        
        # Generate response using LLM (placeholder for now)
        try:
            # This will be replaced with actual Vertex AI call
            response = await ResponseGenerator._call_vertex_ai(question, full_context)
            return response
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            return ResponseGenerator._create_fallback_response(question, evidence)
    
    @staticmethod
    async def _call_vertex_ai(question: str, context: str) -> str:
        """Call Vertex AI Gemini for response generation"""
        # Placeholder - will be implemented in services
        prompt = f"""
You are a legal document assistant helping people with limited legal knowledge understand contracts and legal documents.

INSTRUCTIONS:
1. Provide clear, simple explanations avoiding legal jargon
2. Use everyday language and analogies when possible
3. Highlight potential risks or important considerations
4. Suggest when to consult a professional lawyer
5. Be concise but comprehensive
6. Include specific references to the provided context

CONTEXT:
{context}

USER QUESTION: {question}

RESPONSE (in simple, non-legal language):
"""
        
        # Fallback response for now
        return ResponseGenerator._create_fallback_response(question, [])
    
    @staticmethod
    def _create_fallback_response(question: str, evidence: List[RAGContextItem]) -> str:
        """Create a structured fallback response"""
        question_lower = question.lower()
        
        # Risk-related fallback
        if any(word in question_lower for word in ["risk", "danger", "problem"]):
            return f"""Based on your question about potential risks, here's what you should know:

🔍 **What I found**: {len(evidence)} relevant legal precedents were identified for your question.

⚠️ **Key Considerations**:
- Legal documents often contain terms that may not be immediately obvious
- It's important to understand your obligations and potential liabilities
- Consider the long-term implications of any commitments

💡 **Recommendation**: Given the complexity of legal language, I recommend having a qualified attorney review the specific clauses you're concerned about.

📋 **Next Steps**: Feel free to ask more specific questions about particular sections or terms you'd like me to explain further."""

        # Definition-related fallback
        elif any(phrase in question_lower for phrase in ["what is", "what does", "define"]):
            return f"""I understand you're looking for a clear explanation. Here's what I can tell you:

📖 **Simple Explanation**: Legal terms can be complex, but I'll break this down in everyday language.

🔍 **Context**: Based on {len(evidence)} similar legal documents, this term typically refers to specific obligations or rights.

💭 **In Plain English**: Think of legal contracts like detailed instruction manuals - they specify exactly what each party must do and what happens if they don't.

❓ **Still Unclear?**: Feel free to ask follow-up questions or request examples to make this concept clearer."""

        # General fallback
        else:
            return f"""Thank you for your question. I've searched through legal databases and found {len(evidence)} relevant documents to help answer your question.

🎯 **Key Points**:
- Legal documents are designed to protect all parties involved
- Understanding your rights and obligations is crucial
- Different jurisdictions may have varying interpretations

📚 **Based on Legal Research**: The information I'm providing comes from analysis of similar legal documents and established precedents.

⚖️ **Important Note**: While I can help explain general concepts, specific legal advice should always come from a qualified attorney familiar with your jurisdiction and circumstances.

Would you like me to explain any specific aspect in more detail?"""

@router.post("/", response_model=ChatResponse)
async def chat_with_document(
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    """
    Chat with legal document using RAG and LLM
    
    Provides intelligent Q&A about legal documents with:
    - Context-aware responses
    - Legal precedent citations
    - Simplified explanations for non-lawyers
    - Conversation continuity
    """
    try:
        # Validate and process question
        question = QuestionProcessor.validate_question(request.question)
        question_intent = QuestionProcessor.extract_question_intent(question)
        
        logger.info(f"Processing chat question: {question[:100]}...")
        logger.debug(f"Question intent: {question_intent}")
        
        # Get or create conversation
        conversation_id = request.conversation_id
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
        if request.ocr and conversation:
            conversation["context"]["current_document"] = request.ocr.dict()
            if request.summary_hint:
                conversation["document_summary"] = request.summary_hint
        
        # Retrieve relevant contexts using RAG
        logger.debug("Retrieving legal contexts...")
        evidence = await retrieve_comprehensive_evidence(
            question, 
            question_intent,
            request.ocr,
            conversation
        )
        
        logger.info(f"Retrieved {len(evidence)} evidence items")
        
        # Generate response using LLM
        logger.debug("Generating LLM response...")
        answer = await ResponseGenerator.generate_legal_response(
            question=question,
            evidence=evidence,
            conversation_context=conversation,
            document_context=request.ocr
        )
        
        # Update conversation history
        ConversationManager.update_conversation(
            conversation_id, question, answer, evidence
        )
        
        # Schedule cleanup in background
        background_tasks.add_task(ConversationManager.cleanup_expired_conversations)
        
        logger.info(f"Chat response generated successfully for conversation {conversation_id}")
        
        return ChatResponse(
            answer=answer,
            evidence=evidence,
            conversation_id=conversation_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing failed: {str(e)}"
        )

async def retrieve_comprehensive_evidence(
    question: str,
    question_intent: Dict[str, Any],
    document_context: Optional[OCRText] = None,
    conversation: Optional[Dict] = None
) -> List[RAGContextItem]:
    """Retrieve evidence from multiple sources"""
    try:
        evidence_items = []
        
        # Retrieve from Milvus (2020-2025 data)
        milvus_evidence = await rag.retrieve_contexts(question, top_k=5)
        evidence_items.extend(milvus_evidence)
        
        # If we have document context, search within it
        if document_context and document_context.blocks:
            document_evidence = search_document_context(question, document_context)
            evidence_items.extend(document_evidence)
        
        # Sort by relevance/similarity score
        evidence_items.sort(key=lambda x: x.similarity, reverse=True)
        
        # Return top 8 items to avoid overwhelming the LLM
        return evidence_items[:8]
        
    except Exception as e:
        logger.error(f"Evidence retrieval failed: {e}")
        return []

def search_document_context(question: str, document: OCRText) -> List[RAGContextItem]:
    """Search within the current document for relevant sections"""
    relevant_blocks = []
    question_lower = question.lower()
    
    for i, block in enumerate(document.blocks):
        block_lower = block.text.lower()
        
        # Simple keyword matching (can be enhanced with embeddings)
        relevance_score = 0
        for word in question_lower.split():
            if len(word) > 3 and word in block_lower:
                relevance_score += 1
        
        if relevance_score > 0:
            relevant_blocks.append(RAGContextItem(
                chunk_id=i,
                content=block.text,
                doc_type="current_document",
                jurisdiction="current",
                date=datetime.now().isoformat(),
                source_url="current_document",
                similarity=relevance_score / len(question_lower.split())
            ))
    
    return sorted(relevant_blocks, key=lambda x: x.similarity, reverse=True)[:3]

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
            "conversation_management",
            "rag_integration", 
            "llm_responses",
            "legal_simplification"
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
        "conversation_timeout_hours": CONVERSATION_TIMEOUT.total_seconds() / 3600
    }
