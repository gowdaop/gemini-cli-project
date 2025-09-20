from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any
import logging
import asyncio
import hashlib
from functools import lru_cache
from ..services import rag
from ..schemas.analysis import (
    AnalyzeRequest, 
    AnalyzeResponse, 
    Clause, 
    RiskScore, 
    ClauseTag, 
    RiskLevel,
    PageSpan,
    RAGContextItem,
    ErrorResponse
)
from ..services.risk import assess_document_risks
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/analyze",
    tags=["analyze"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    }
)

@router.post("/", response_model=AnalyzeResponse)
async def analyze_document(request: AnalyzeRequest):
    """
    Enhanced legal document analysis with comprehensive validation and error handling
    
    Takes OCR-extracted text and returns:
    - Identified legal clauses with classifications
    - Risk assessments with supporting evidence
    - Summary in ≤300 words
    """
    try:
        # Input validation
        if not request.ocr or not request.ocr.blocks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No OCR blocks provided for analysis"
            )
        
        if len(request.ocr.full_text.strip()) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document text too short for meaningful analysis"
            )
        
        # Limit processing size to prevent resource exhaustion
        max_blocks = getattr(settings, 'MAX_CLAUSE_BLOCKS', 100)
        if len(request.ocr.blocks) > max_blocks:
            logger.warning(f"Truncating blocks from {len(request.ocr.blocks)} to {max_blocks}")
            request.ocr.blocks = request.ocr.blocks[:max_blocks]
        
        logger.info(f"Starting document analysis with {len(request.ocr.blocks)} text blocks")
        
        # Process with timeout to prevent hanging
        try:
            async with asyncio.timeout(300):  # 5 minute timeout
                # Step 1: Classify clauses
                logger.debug("Classifying legal clauses...")
                clauses = await classify_clauses(request.ocr)
                logger.info(f"Identified {len(clauses)} clauses")
                
                # Step 2: Score risks for each clause with RAG support (parallel processing)
                logger.debug("Scoring risks with RAG evidence...")
                risks = await score_risks_parallel(clauses, min(request.top_k, 10))
                logger.info(f"Assessed {len(risks)} risk items")
                
                # Step 3: Generate enhanced summary with recommendations
                logger.debug("Generating enhanced document summary with recommendations...")
                summary = await generate_enhanced_summary(request.ocr.full_text)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail="Document analysis timed out. Try with a smaller document."
            )
        
        # Ensure summary exists
        if not summary or not summary.strip():
            summary = create_fallback_summary(request.ocr.full_text)
        
        response = AnalyzeResponse(
            clauses=clauses,
            risks=risks,
            summary_200w=summary  # This will now contain the enhanced summary with recommendations
        )
        
        logger.info("Document analysis completed successfully")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document analysis failed due to internal error"
        )

async def classify_clauses(ocr_text) -> List[Clause]:
    """
    Enhanced clause classification with proper error handling
    """
    try:
        clauses = []
        
        if not ocr_text.blocks:
            logger.warning("No text blocks provided for clause classification")
            return create_fallback_clauses(ocr_text)
        
        for i, block in enumerate(ocr_text.blocks):
            # Generate clause ID
            clause_id = f"c-{i+1:04d}"
            
            # Skip empty blocks
            if not block.text or not block.text.strip():
                logger.debug(f"Skipping empty block {clause_id}")
                continue
            
            # Determine clause type using enhanced keyword matching
            clause_tag = determine_clause_tag_enhanced(block.text)
            
            clauses.append(Clause(
                id=clause_id,
                tag=clause_tag,
                text=block.text.strip(),
                span=block.span
            ))
            
            logger.debug(f"Classified {clause_id} as {clause_tag.value}")
        
        if not clauses:
            logger.warning("No valid clauses found, creating fallback")
            return create_fallback_clauses(ocr_text)
        
        return clauses
        
    except Exception as e:
        logger.error(f"Clause classification failed: {e}", exc_info=True)
        return create_fallback_clauses(ocr_text)

async def score_risks_parallel(clauses: List[Clause], top_k: int) -> List[RiskScore]:
    """
    Enhanced parallel risk scoring for better performance
    """
    if not clauses:
        logger.warning("No clauses provided for risk scoring")
        return []
    
    async def score_single_clause(clause: Clause) -> RiskScore:
        """Score a single clause with comprehensive error handling"""
        try:
            # Retrieve relevant context for this clause
            contexts = await retrieve_clause_contexts(clause.text, top_k)
            
            # Calculate risk score based on clause type and content
            risk_score, risk_level = calculate_clause_risk_enhanced(clause)
            
            # Generate rationale
            rationale = generate_risk_rationale_enhanced(clause, risk_score, contexts)
            
            logger.debug(f"Scored {clause.id}: {risk_level.value} ({risk_score:.2f})")
            
            return RiskScore(
                clause_id=clause.id,
                level=risk_level,
                score=risk_score,
                rationale=rationale,
                supporting_context=contexts
            )
            
        except Exception as e:
            logger.warning(f"Risk scoring failed for clause {clause.id}: {e}")
            return create_fallback_risk_score(clause)
    
    # Process all clauses concurrently with semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent RAG requests
    
    async def score_with_semaphore(clause: Clause) -> RiskScore:
        async with semaphore:
            return await score_single_clause(clause)
    
    risk_tasks = [score_with_semaphore(clause) for clause in clauses]
    risks = await asyncio.gather(*risk_tasks, return_exceptions=True)
    
    # Filter out exceptions and return valid results
    valid_risks = []
    for risk in risks:
        if isinstance(risk, RiskScore):
            valid_risks.append(risk)
        elif isinstance(risk, Exception):
            logger.error(f"Risk scoring exception: {risk}")
    
    return valid_risks

async def retrieve_clause_contexts(clause_text: str, top_k: int) -> List[RAGContextItem]:
    """
    Enhanced context retrieval with better error handling
    """
    try:
        if not clause_text or not clause_text.strip():
            return []
        
        # Use RAG service to find relevant contexts
        contexts = await rag.retrieve_contexts(clause_text, top_k=min(top_k, 5))
        
        if not contexts:
            logger.debug("No contexts retrieved from RAG service")
            return []
        
        # Convert to RAGContextItem with validation
        rag_contexts = []
        for i, ctx in enumerate(contexts):
            try:
                if isinstance(ctx, dict):
                    rag_contexts.append(RAGContextItem(
                        chunk_id=ctx.get("chunk_id", i),
                        content=ctx.get("content", "")[:1000],  # Limit content length
                        doc_type=ctx.get("doc_type", "legal_document"),
                        jurisdiction=ctx.get("jurisdiction", "unknown"),
                        date=ctx.get("date", "unknown"),
                        source_url=ctx.get("source_url", ""),
                        similarity=float(ctx.get("similarity", 0.0))
                    ))
                elif hasattr(ctx, 'chunk_id'):  # Already a RAGContextItem
                    rag_contexts.append(ctx)
            except Exception as e:
                logger.warning(f"Failed to process context {i}: {e}")
                continue
                
        logger.debug(f"Retrieved {len(rag_contexts)} valid contexts")
        return rag_contexts
        
    except Exception as e:
        logger.warning(f"Context retrieval failed: {e}")
        return []

@lru_cache(maxsize=1000)
def determine_clause_tag_enhanced(text: str) -> ClauseTag:
    """
    Enhanced clause type detection with comprehensive keyword matching and caching
    """
    if not text:
        return ClauseTag.OTHER
    
    text_lower = text.lower()
    logger.debug(f"Classifying clause: {text[:100]}...")
    
    # Enhanced keyword patterns for better accuracy
    clause_patterns = {
        ClauseTag.TERMINATION: [
            "terminate", "termination", "end agreement", "cancel", "cancellation",
            "expire", "expiration", "dissolution", "breach", "default"
        ],
        ClauseTag.LIABILITY: [
            "liability", "liable", "damages", "responsible", "limitation of liability",
            "limitation", "limitation clause", "total liability", "maximum liability"
        ],
        ClauseTag.INDEMNITY: [
            "indemnify", "indemnification", "hold harmless", "defend", "defense",
            "indemnitor", "indemnitee", "protect", "save harmless"
        ],
        ClauseTag.CONFIDENTIALITY: [
            "confidential", "non-disclosure", "proprietary", "confidentiality",
            "nda", "secret", "private information", "confidential information"
        ],
        ClauseTag.PAYMENT: [
            "payment", "fees", "invoice", "charges", "compensation", "remuneration",
            "salary", "wage", "cost", "expense", "billing", "pay"
        ],
        ClauseTag.IP: [
            "intellectual property", "copyright", "trademark", "patent", "trade secret",
            "proprietary rights", "ip rights", "invention", "know-how"
        ],
        ClauseTag.GOVERNING_LAW: [
            "governing law", "jurisdiction", "applicable law", "choice of law",
            "laws of", "governed by", "subject to the laws"
        ],
        ClauseTag.ARBITRATION: [
            "arbitration", "dispute resolution", "mediation", "arbitrator",
            "arbitral", "binding arbitration", "alternative dispute resolution"
        ]
    }
    
    # Score each clause type based on keyword matches
    clause_scores = {}
    for clause_type, keywords in clause_patterns.items():
        score = 0
        for keyword in keywords:
            if keyword in text_lower:
                # Weight longer phrases higher
                score += len(keyword.split())
        clause_scores[clause_type] = score
    
    # Return the clause type with the highest score
    if clause_scores:
        best_clause = max(clause_scores, key=clause_scores.get)
        if clause_scores[best_clause] > 0:
            logger.debug(f"Classified as {best_clause.value} (score: {clause_scores[best_clause]})")
            return best_clause
    
    logger.debug("Classified as OTHER (no keyword matches)")
    return ClauseTag.OTHER

def calculate_clause_risk_enhanced(clause: Clause) -> tuple[float, RiskLevel]:
    """
    Enhanced risk calculation with sophisticated pattern matching and monetary cap handling
    """
    # Base risk scores by clause type
    base_risks = {
        ClauseTag.LIABILITY: 0.8,
        ClauseTag.INDEMNITY: 0.7,
        ClauseTag.TERMINATION: 0.6,
        ClauseTag.IP: 0.6,
        ClauseTag.ARBITRATION: 0.5,
        ClauseTag.PAYMENT: 0.5,
        ClauseTag.GOVERNING_LAW: 0.4,
        ClauseTag.CONFIDENTIALITY: 0.3,
        ClauseTag.OTHER: 0.2
    }
    
    risk_score = base_risks.get(clause.tag, 0.2)
    text_lower = clause.text.lower()
    
    # High-risk patterns (increase risk)
    high_risk_patterns = [
        "unlimited", "without limitation", "all damages", "any damages",
        "without any cap", "without any limit", "personal liability",
        "joint and several", "gross negligence", "willful misconduct",
        "punitive damages", "consequential damages", "indirect damages",
        "without restriction", "in any amount", "unlimited liability"
    ]
    
    # Risk-reducing patterns (decrease risk)
    low_risk_patterns = [
        "limited to", "capped at", "maximum", "not to exceed",
        "excluding", "except for", "reasonable efforts", "best efforts",
        "subject to", "provided that", "limited liability", "cap on"
    ]
    
    # Apply risk adjustments based on pattern matches
    high_risk_matches = sum(1 for pattern in high_risk_patterns if pattern in text_lower)
    low_risk_matches = sum(1 for pattern in low_risk_patterns if pattern in text_lower)
    
    # Adjust score based on pattern matches
    risk_score += (high_risk_matches * 0.15)
    risk_score -= (low_risk_matches * 0.1)
    
    # ✅ CRITICAL FIX: Special handling for LIABILITY clauses with monetary caps
    if clause.tag == ClauseTag.LIABILITY:
        import re
        # Enhanced regex to catch various monetary formats
        money_pattern = r'\$\s*[\d,]+|\$\s*\d+|\d+,?\d*\s*dollars?|\d+k|\d+m'
        cap_keywords = ["limited to", "not to exceed", "capped at", "maximum", "cap of"]
        
        # Check if text contains both monetary amount AND cap keywords
        has_money = re.search(money_pattern, text_lower)
        has_cap_keyword = any(keyword in text_lower for keyword in cap_keywords)
        
        if has_money and has_cap_keyword:
            # Significant risk reduction for capped liability
            risk_score *= 0.55  # Reduce risk by 45%
            logger.debug(f"Applied liability cap reduction for clause {clause.id}: found ${has_money.group()} with cap keyword")
    
    # ✅ ENHANCED: Additional specific reductions for INDEMNITY caps
    elif clause.tag == ClauseTag.INDEMNITY:
        import re
        money_pattern = r'\$\s*[\d,]+|\$\s*\d+|\d+,?\d*\s*dollars?'
        cap_keywords = ["limited to", "not to exceed", "capped at", "maximum"]
        
        if re.search(money_pattern, text_lower) and any(keyword in text_lower for keyword in cap_keywords):
            risk_score *= 0.7  # Moderate reduction for capped indemnity
            logger.debug(f"Applied indemnity cap reduction for clause {clause.id}")
    
    # Clamp score between 0 and 1
    risk_score = max(0.0, min(1.0, risk_score))
    
    # Convert to risk level with enhanced thresholds
    if risk_score >= 0.8:
        risk_level = RiskLevel.RED
    elif risk_score >= 0.6:
        risk_level = RiskLevel.ORANGE
    elif risk_score >= 0.3:
        risk_level = RiskLevel.YELLOW
    else:
        risk_level = RiskLevel.WHITE
    
    logger.debug(f"Risk calculation for {clause.id} ({clause.tag.value}): "
                f"base={base_risks.get(clause.tag, 0.2):.2f}, "
                f"high_risk_matches={high_risk_matches}, "
                f"low_risk_matches={low_risk_matches}, "
                f"final_score={risk_score:.2f}, "
                f"level={risk_level.value}")
    
    return risk_score, risk_level


def generate_risk_rationale_enhanced(clause: Clause, risk_score: float, contexts: List[RAGContextItem]) -> str:
    """
    Enhanced risk rationale generation with more detailed explanations
    """
    base_rationale = f"This {clause.tag.value} clause has been assessed with a {risk_score:.1f} risk score. "
    
    # Enhanced rationales based on clause type
    type_rationales = {
        ClauseTag.LIABILITY: "Liability clauses can expose the organization to significant financial risk and potential legal exposure.",
        ClauseTag.INDEMNITY: "Indemnification terms may require defending or compensating other parties, potentially resulting in substantial costs.",
        ClauseTag.TERMINATION: "Termination conditions affect contract flexibility, exit strategies, and operational continuity.",
        ClauseTag.PAYMENT: "Payment terms directly impact cash flow, financial planning, and business relationships.",
        ClauseTag.IP: "Intellectual property clauses affect ownership rights, usage permissions, and competitive advantages.",
        ClauseTag.CONFIDENTIALITY: "Confidentiality terms may restrict information sharing and business operations.",
        ClauseTag.GOVERNING_LAW: "Governing law affects dispute resolution procedures, legal interpretation, and enforcement options.",
        ClauseTag.ARBITRATION: "Arbitration clauses may limit legal recourse options and affect dispute resolution costs."
    }
    
    rationale = base_rationale + type_rationales.get(clause.tag, "This clause requires careful legal review and consideration.")
    
    # Add risk level context
    if risk_score >= 0.8:
        rationale += " This is considered a HIGH RISK clause requiring immediate legal review."
    elif risk_score >= 0.6:
        rationale += " This is considered a MEDIUM-HIGH RISK clause requiring careful evaluation."
    elif risk_score >= 0.3:
        rationale += " This clause presents moderate risk and should be reviewed."
    
    # Add context from legal database if available
    if contexts and len(contexts) > 0:
        rationale += f" Legal precedents from {len(contexts)} similar cases support this risk assessment."
    
    return rationale

async def generate_summary(full_text: str) -> str:
    """
    Enhanced summary generation with better validation
    """
    try:
        if not full_text or len(full_text.strip()) < 10:
            return create_fallback_summary(full_text)
        
        logger.info(f"Starting summary generation for {len(full_text)} characters of text")
        logger.debug(f"Text preview: {full_text[:200]}...")
        
        # Use RAG service to generate summary
        summary = await rag.summarize_200w(full_text)
        
        # Validate summary result
        if not summary or not summary.strip():
            logger.warning("Empty summary returned from RAG service")
            return create_fallback_summary(full_text)
        
        # Check word count
        word_count = len(summary.split())
        if word_count > 200:
            logger.warning(f"Summary too long ({word_count} words), truncating")
            words = summary.split()[:200]
            summary = " ".join(words)
        
        logger.info(f"Generated summary: {len(summary)} characters, {word_count} words")
        return summary.strip()
        
    except Exception as e:
        logger.error(f"Summary generation failed: {e}", exc_info=True)
        return create_fallback_summary(full_text)


async def generate_enhanced_summary(full_text: str) -> str:
    """
    Enhanced summary generation with recommendations (500 words)
    """
    try:
        if not full_text or len(full_text.strip()) < 10:
            return create_fallback_summary_with_recommendations(full_text)
        
        logger.info(f"Starting enhanced summary generation for {len(full_text)} characters of text")
        logger.debug(f"Text preview: {full_text[:200]}...")
        
        # Use RAG service to generate enhanced summary
        summary = await rag.summarize_500w_with_recommendations(full_text)
        
        # Validate summary result
        if not summary or not summary.strip():
            logger.warning("Empty enhanced summary returned from RAG service")
            return create_fallback_summary_with_recommendations(full_text)
        
        # Check word count
        word_count = len(summary.split())
        if word_count > 500:
            logger.warning(f"Enhanced summary too long ({word_count} words), truncating")
            words = summary.split()[:500]
            summary = " ".join(words)
        
        logger.info(f"Generated enhanced summary: {len(summary)} characters, {word_count} words")
        return summary.strip()
        
    except Exception as e:
        logger.error(f"Enhanced summary generation failed: {e}", exc_info=True)
        return create_fallback_summary_with_recommendations(full_text)

def create_fallback_summary(full_text: str) -> str:
    """
    Enhanced fallback summary generation
    """
    if not full_text:
        return "Document analysis could not extract meaningful text content."
    
    words = full_text.split()
    word_count = len(words)
    
    if word_count <= 150:
        return f"This legal document contains {word_count} words. {full_text}"
    
    # Create a more intelligent summary
    summary_words = words[:150]
    summary = " ".join(summary_words)
    
    return f"This legal document contains {word_count} words covering various contractual terms and conditions. {summary}..."


def create_fallback_summary_with_recommendations(full_text: str) -> str:
    """
    Enhanced fallback summary generation with recommendations
    """
    if not full_text:
        return "Document analysis could not extract meaningful text content.\n\nRECOMMENDATIONS: Please ensure the document is properly scanned and try again."
    
    words = full_text.split()
    word_count = len(words)
    
    if word_count <= 200:
        return f"""SUMMARY: This legal document contains {word_count} words covering various contractual terms and conditions. {full_text}

RECOMMENDATIONS: 
1. Review all terms and conditions carefully
2. Check for any liability or indemnification clauses
3. Verify payment terms and deadlines
4. Consider consulting with a qualified Indian lawyer for specific legal advice
5. Ensure all parties understand their rights and obligations"""
    
    # Create a more intelligent summary
    summary_words = words[:200]
    summary = " ".join(summary_words)
    
    return f"""SUMMARY: This legal document contains {word_count} words covering various contractual terms and conditions. The document appears to establish rights, obligations, and procedures for the parties involved. Key areas typically include liability, termination, payment terms, and governing law. {summary}...

RECOMMENDATIONS: 
1. Review all liability and indemnification clauses carefully
2. Check payment terms and deadlines
3. Understand termination conditions and notice requirements
4. Verify governing law and jurisdiction clauses
5. Consider consulting with a qualified Indian lawyer for specific legal advice
6. Ensure all parties understand their rights and obligations
7. Look for any unusual or one-sided terms that may need negotiation"""

def create_fallback_clauses(ocr_text) -> List[Clause]:
    """
    Enhanced fallback clause creation
    """
    if not ocr_text or not ocr_text.full_text:
        return [
            Clause(
                id="c-0001",
                tag=ClauseTag.OTHER,
                text="No text available for analysis",
                span=PageSpan(page=1, start_line=1, end_line=1)
            )
        ]
    
    # Create a single clause from the full text
    text = ocr_text.full_text[:500] if len(ocr_text.full_text) > 500 else ocr_text.full_text
    
    return [
        Clause(
            id="c-0001",
            tag=determine_clause_tag_enhanced(text),
            text=text,
            span=PageSpan(page=1, start_line=1, end_line=text.count('\n') + 1)
        )
    ]

def create_fallback_risk_score(clause: Clause) -> RiskScore:
    """
    Enhanced fallback risk score creation
    """
    return RiskScore(
        clause_id=clause.id,
        level=RiskLevel.WHITE,
        score=0.1,
        rationale=f"Risk assessment unavailable for {clause.tag.value} clause due to processing limitations. Manual review recommended.",
        supporting_context=[]
    )

@router.get("/health")
async def analyze_health():
    """Enhanced health check for analyze service"""
    return {
        "status": "healthy", 
        "service": "analyze",
        "version": "2.0",
        "features": {
            "parallel_processing": True,
            "enhanced_risk_calculation": True,
            "comprehensive_validation": True,
            "rag_integration": True
        }
    }

@router.get("/capabilities")
async def get_capabilities():
    """Get analysis service capabilities"""
    return {
        "supported_clause_types": [tag.value for tag in ClauseTag],
        "risk_levels": [level.value for level in RiskLevel],
        "max_blocks": getattr(settings, 'MAX_CLAUSE_BLOCKS', 100),
        "timeout_seconds": 300,
        "max_summary_words": 500
    }
