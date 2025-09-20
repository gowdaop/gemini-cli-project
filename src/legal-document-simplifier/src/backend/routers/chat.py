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
    OCRText,
    Clause,
    RiskScore
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
            "question_type_history": [],  # Track question types for better routing
            "analysis_clauses": {},  # {clause_id: clause_dict}
            "analysis_risks": {}     # {clause_id: risk_dict}
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
    def store_analysis_results(conversation_id: str, clauses: List[Clause], risks: List[RiskScore]) -> None:
        """Store analysis results for later reference"""
        conversation = conversation_store.get(conversation_id)
        if not conversation:
            logger.warning(f"Conversation {conversation_id} not found for storing analysis results")
            return
        
        # Store clauses by ID - handle both Pydantic objects and dictionaries
        conversation["analysis_clauses"] = {}
        for clause in clauses:
            if hasattr(clause, 'id'):  # Pydantic object
                clause_id = clause.id
                clause_tag = clause.tag.value if hasattr(clause.tag, 'value') else str(clause.tag)
                clause_text = clause.text
                clause_span = clause.span.dict() if clause.span else None
            else:  # Dictionary
                clause_id = clause.get('id')
                clause_tag = clause.get('tag')
                clause_text = clause.get('text')
                clause_span = clause.get('span')
            
            conversation["analysis_clauses"][clause_id] = {
                "id": clause_id,
                "tag": clause_tag,
                "text": clause_text,
                "span": clause_span
            }
        
        # Store risks by clause ID - handle both Pydantic objects and dictionaries
        conversation["analysis_risks"] = {}
        for risk in risks:
            if hasattr(risk, 'clause_id'):  # Pydantic object
                risk_clause_id = risk.clause_id
                risk_level = risk.level.value if hasattr(risk.level, 'value') else str(risk.level)
                risk_score = risk.score
                risk_rationale = risk.rationale
                risk_context = [ctx.dict() for ctx in risk.supporting_context] if risk.supporting_context else []
            else:  # Dictionary
                risk_clause_id = risk.get('clause_id')
                risk_level = risk.get('level')
                risk_score = risk.get('score')
                risk_rationale = risk.get('rationale')
                risk_context = risk.get('supporting_context', [])
            
            conversation["analysis_risks"][risk_clause_id] = {
                "clause_id": risk_clause_id,
                "level": risk_level,
                "score": risk_score,
                "rationale": risk_rationale,
                "supporting_context": risk_context
            }
        
        conversation["last_activity"] = datetime.utcnow()
        logger.info(f"Stored {len(clauses)} clauses and {len(risks)} risks for conversation {conversation_id}")
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
        summary_hint: Optional[str] = None,
        selected_clause_id: Optional[str] = None,
        selected_risk_level: Optional[str] = None
    ) -> Dict[str, Any]:
        """Enhanced chat request handling with document-first routing"""
        
        if not self.initialized:
            await self.initialize()
        
        original_question = question  # Keep original for history
        question = EnhancedQuestionProcessor.validate_question(question)
        
        if selected_risk_level:
            selected_risk_level = selected_risk_level.lower()
        
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
        
        question_intent = EnhancedQuestionProcessor.extract_question_intent(
            question, conversation
        )
        
        logger.info(f"Processing question (type: {question_intent['type']}, confidence: {question_intent['confidence']:.2f})")
        
        if ocr and conversation:
            conversation["context"]["current_document"] = ocr.dict()
            if summary_hint:
                conversation["document_summary"] = summary_hint
        
        evidence, question_for_llm = await self._retrieve_intelligent_evidence(
            question,
            question_intent,
            ocr,
            conversation,
            selected_clause_id,
            selected_risk_level
        )
        
        logger.info(f"Retrieved {len(evidence)} evidence items for LLM.")
        
        answer = await rag.answer_with_vertex(
            question=question_for_llm,
            contexts=evidence,
            summary_hint=conversation.get("document_summary")
        )
        
        ConversationManager.update_conversation(
            conversation_id, original_question, answer, evidence, question_intent["type"]
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
        conversation: Optional[Dict] = None,
        selected_clause_id: Optional[str] = None,
        selected_risk_level: Optional[str] = None
    ) -> tuple[List[RAGContextItem], str]:
        """Document-first evidence retrieval. Returns evidence and the question for the LLM."""
        try:
            evidence_items = []
            selected_context = None
            
            if selected_risk_level:
                selected_risk_level = selected_risk_level.lower()

            has_analysis = conversation and (conversation.get("analysis_clauses") or conversation.get("analysis_risks"))
            has_selection = selected_clause_id or selected_risk_level
            has_ocr = document_context and document_context.blocks and any(b.text.strip() for b in document_context.blocks)

            if has_analysis and has_selection:
                selected_context = self._get_selected_context(
                    conversation, selected_clause_id, selected_risk_level
                )
                if selected_context:
                    context_evidence = self._create_context_evidence(selected_context)
                    evidence_items.extend(context_evidence)
                    question = self._enhance_question_with_context(question, selected_context)

            if has_ocr or selected_context:
                logger.info("route=document_first")
                document_evidence = []
                if has_ocr:
                    document_evidence = self._search_document_context(question, document_context)
                    evidence_items.extend(document_evidence)
                
                logger.info(f"doc_blocks={len(document_evidence)}")
                logger.info("use_web_fallback=False")
                
                legal_evidence = await rag.retrieve_contexts(
                    question, 
                    top_k=5,
                    use_web_fallback=False
                )
                evidence_items.extend(legal_evidence)
            else:
                logger.info("route=generic")
                logger.info("doc_blocks=0")
                logger.info("use_web_fallback=True")
                
                generic_evidence = await rag.retrieve_contexts(
                    question, 
                    top_k=6, 
                    use_web_fallback=True
                )
                evidence_items.extend(generic_evidence)
            
            seen_evidence = set()
            unique_evidence = []
            for item in evidence_items:
                evidence_key = f"{item.doc_type}:{hashlib.md5(item.content[:200].encode()).hexdigest()}"
                if evidence_key not in seen_evidence:
                    seen_evidence.add(evidence_key)
                    unique_evidence.append(item)
            
            unique_evidence.sort(key=lambda x: x.similarity, reverse=True)
            final_evidence = unique_evidence[:8]
            
            logger.info(f"Compiled {len(final_evidence)} unique evidence items.")
            return final_evidence, question
            
        except Exception as e:
            logger.error(f"Intelligent evidence retrieval failed: {e}", exc_info=True)
            try:
                evidence = await rag.retrieve_contexts(question, top_k=5, use_web_fallback=True)
                return evidence, question
            except:
                return [], question
            
    def _get_selected_context(
        self, 
        conversation: Dict, 
        clause_id: Optional[str], 
        risk_level: Optional[str]
    ) -> Dict[str, Any]:
        """Extract selected clause/risk context with normalized risk levels"""
        context = {}
        
        if not conversation:
            return context
        
        if clause_id and clause_id in conversation.get("analysis_clauses", {}):
            context["selected_clause"] = conversation["analysis_clauses"][clause_id]
            if clause_id in conversation.get("analysis_risks", {}):
                context["selected_risk"] = conversation["analysis_risks"][clause_id]
        
        if risk_level:
            risk_level_lower = risk_level.lower()
            matching_risks = {
                cid: risk for cid, risk in conversation.get("analysis_risks", {}).items()
                if risk.get("level", "").lower() == risk_level_lower
            }
            context["filtered_risks"] = matching_risks
            context["filtered_clauses"] = {
                cid: conversation["analysis_clauses"].get(cid)
                for cid in matching_risks.keys()
                if cid in conversation.get("analysis_clauses", {})
            }
        
        return context

    def _search_document_context(self, question: str, document: OCRText) -> List[RAGContextItem]:
        """Enhanced document context search returning top 5 blocks"""
        relevant_blocks = []
        question_lower = question.lower()
        question_words = [word for word in question_lower.split()]
        
        for i, block in enumerate(document.blocks):
            block_lower = block.text.lower()
            relevance_score = 0
            question_phrases = [' '.join(question_words[i:i+2]) for i in range(len(question_words)-1)]
            for phrase in question_phrases:
                if phrase in block_lower:
                    relevance_score += 2
            for word in question_words:
                if word in block_lower:
                    relevance_score += 1
            context_indicators = ["clause", "section", "agreement", "shall", "party", "terms"]
            for indicator in context_indicators:
                if indicator in block_lower:
                    relevance_score += 0.5
            
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
        
        return sorted(relevant_blocks, key=lambda x: x.similarity, reverse=True)[:5]
    
    def _get_source_summary(self, evidence: List[RAGContextItem]) -> Dict[str, int]:
        """Get summary of sources used"""
        sources = {}
        for item in evidence:
            source_type = item.doc_type
            sources[source_type] = sources.get(source_type, 0) + 1
        return sources

    def _create_context_evidence(self, selected_context: Dict[str, Any]) -> List[RAGContextItem]:
        """Create evidence from selected context"""
        evidence = []
        
        if "selected_clause" in selected_context:
            clause = selected_context["selected_clause"]
            clause_content = f"""SELECTED TERMINATION CLAUSE ANALYSIS:

Clause Type: {clause['tag'].upper()}
Clause Text: "{clause['text']}"
Clause ID: {clause['id']}

This is the specific clause the user has selected for analysis. The clause deals with termination and notice requirements in the contract."""
            
            evidence.append(RAGContextItem(
                chunk_id=999001,
                content=clause_content,
                doc_type="selected_clause",
                jurisdiction="current_document",
                date=datetime.now().isoformat()[:10],
                source_url="current_analysis",
                similarity=1.0
            ))
        
        if "selected_risk" in selected_context:
            risk = selected_context["selected_risk"]
            risk_content = f"""RISK ASSESSMENT FOR SELECTED CLAUSE:

Risk Level: {risk['level'].upper()}
Risk Score: {risk['score']:.2f}/1.0
Rationale: {risk['rationale']}

This risk assessment applies specifically to the selected termination clause."""
            
            evidence.append(RAGContextItem(
                chunk_id=999002,
                content=risk_content,
                doc_type="risk_assessment",
                jurisdiction="current_analysis", 
                date=datetime.now().isoformat()[:10],
                source_url="current_analysis",
                similarity=0.95
            ))
            
            for i, ctx in enumerate(risk.get("supporting_context", [])):
                evidence.append(RAGContextItem(**ctx))
        
        if "filtered_clauses" in selected_context:
            for i, (clause_id, clause) in enumerate(selected_context["filtered_clauses"].items()):
                evidence.append(RAGContextItem(
                    chunk_id=999100 + i,
                    content=f"Related {clause['tag']} clause: {clause['text']}",
                    doc_type="related_clause",
                    jurisdiction="current_document",
                    date=datetime.now().isoformat()[:10],
                    source_url="current_analysis",
                    similarity=0.9
                ))
        
        if "filtered_risks" in selected_context:
            for i, (clause_id, risk) in enumerate(selected_context["filtered_risks"].items()):
                evidence.append(RAGContextItem(
                    chunk_id=999200 + i,
                    content=f"Related {risk['level']} risk: {risk['rationale']}",
                    doc_type="related_risk",
                    jurisdiction="current_analysis",
                    date=datetime.now().isoformat()[:10],
                    source_url="current_analysis",
                    similarity=0.85
                ))
        
        return evidence

    def _enhance_question_with_context(self, question: str, selected_context: Dict[str, Any]) -> str:
        """Make question more specific based on selected context"""
        if "selected_clause" in selected_context:
            clause = selected_context["selected_clause"]
            clause_type = clause.get("tag", "clause").replace("_", " ")
            clause_text = clause["text"]
            
            enhanced_question = f"""The user has selected a specific '{clause_type}' clause from their document for analysis.

SELECTED CLAUSE TEXT:
"{clause_text}"

Based on this specific clause, the user's question is: "{question}"

Please provide a detailed analysis of THIS SPECIFIC clause, explaining its implications, requirements, and any potential risks or considerations based on the text provided."""
            
            return enhanced_question

        elif "filtered_risks" in selected_context and selected_context["filtered_risks"]:
            risk_level = next(iter(selected_context["filtered_risks"].values())).get('level', 'Unknown')
            num_risks = len(selected_context["filtered_risks"])
            
            clauses_text = []
            if "filtered_clauses" in selected_context:
                for clause in selected_context["filtered_clauses"].values():
                    clauses_text.append(f"- {clause.get('tag', 'Clause')}: \"{clause.get('text', '')[:100]}...\"")
            
            clauses_summary = "\n".join(clauses_text)

            enhanced_question = f"""The user has filtered the document to focus on all items with a '{risk_level.upper()}' risk level. There are {num_risks} such item(s).

The associated clauses are:
{clauses_summary}

The user's question is: \"{question}\" 

Please provide an answer based on the context of these '{risk_level.upper()}' risk items and their associated clauses."""
            
            return enhanced_question

        return question

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
            summary_hint=request.summary_hint,
            selected_clause_id=request.selected_clause_id,
            selected_risk_level=request.selected_risk_level
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
        "analysis_clauses": conversation.get("analysis_clauses", {}),
        "analysis_risks": conversation.get("analysis_risks", {}),
        "analytics": {
            "question_types": dict(zip(*np.unique(question_types, return_counts=True))) if question_types else {},
            "source_usage": source_usage,
            "has_document_context": "current_document" in conversation.get("context", {}),
            "has_analysis_data": bool(conversation.get("analysis_clauses") or conversation.get("analysis_risks"))
        }
    }
@router.post("/store-analysis/{conversation_id}")
async def store_analysis_results(
    conversation_id: str,
    clauses: List[Clause],
    risks: List[RiskScore]
):
    """Store analysis results for conversation context"""
    ConversationManager.store_analysis_results(conversation_id, clauses, risks)
    return {"message": "Analysis results stored", "clauses": len(clauses), "risks": len(risks)}

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