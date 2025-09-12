from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class ClauseTag(str, Enum):
    TERMINATION = "termination"
    LIABILITY = "liability"
    INDEMNITY = "indemnity"
    CONFIDENTIALITY = "confidentiality"
    PAYMENT = "payment"
    IP = "ip"
    GOVERNING_LAW = "governing_law"
    ARBITRATION = "arbitration"
    OTHER = "other"

class RiskLevel(str, Enum):
    WHITE = "white"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"

class PageSpan(BaseModel):
    """Represents the location of text within a document"""
    page: int = Field(..., ge=1, description="Page number (1-indexed)")
    start_line: int = Field(..., ge=1, description="Starting line number")
    end_line: int = Field(..., ge=1, description="Ending line number")

class OCRBlock(BaseModel):
    """A block of text extracted from the document"""
    text: str = Field(..., description="The extracted text content")
    span: PageSpan = Field(..., description="Location information for highlighting")

class OCRText(BaseModel):
    """Complete OCR result from document processing"""
    full_text: str = Field(..., description="Complete extracted text")
    blocks: List[OCRBlock] = Field(default_factory=list, description="Structured text blocks with positions")

class RAGContextItem(BaseModel):
    """Context item retrieved from vector search"""
    chunk_id: int = Field(..., description="Unique identifier for the chunk")
    content: str = Field(..., description="The retrieved text content")
    doc_type: str = Field(..., description="Type of legal document")
    jurisdiction: str = Field(..., description="Legal jurisdiction")
    date: str = Field(..., description="Document date")
    source_url: str = Field(..., description="Source URL or reference")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score")

class Clause(BaseModel):
    """Identified legal clause"""
    id: str = Field(..., description="Unique clause identifier")
    tag: ClauseTag = Field(..., description="Type of legal clause")
    text: str = Field(..., description="The clause text")
    span: PageSpan = Field(..., description="Location in document")

class RiskScore(BaseModel):
    """Risk assessment for a clause"""
    clause_id: str = Field(..., description="Reference to clause ID")
    level: RiskLevel = Field(..., description="Risk level classification")
    score: float = Field(..., ge=0.0, le=1.0, description="Numerical risk score")
    rationale: str = Field(..., description="Explanation of the risk assessment")
    supporting_context: List[RAGContextItem] = Field(default_factory=list, description="Evidence from legal database")
    recommendations: List[str] = []

# Request/Response Models
class AnalyzeRequest(BaseModel):
    """Request for document analysis"""
    ocr: OCRText = Field(..., description="OCR result to analyze")
    top_k: int = Field(8, ge=1, le=50, description="Number of context items to retrieve")

class AnalyzeResponse(BaseModel):
    """Complete document analysis result"""
    clauses: List[Clause] = Field(..., description="Identified legal clauses")
    risks: List[RiskScore] = Field(..., description="Risk assessments")
    summary_200w: Optional[str] = Field(None, description="Summary in ≤200 words")

class UploadResponse(BaseModel):
    """Response from document upload"""
    ocr: OCRText = Field(..., description="OCR extraction result")

class ChatRequest(BaseModel):
    """Request for document Q&A"""
    question: str = Field(..., min_length=1, description="User's question about the document")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")
    ocr: Optional[OCRText] = Field(None, description="Document context")
    summary_hint: Optional[str] = Field(None, description="Additional context hint")

class ChatResponse(BaseModel):
    """Response from document Q&A"""
    answer: str = Field(..., description="AI-generated answer")
    evidence: List[RAGContextItem] = Field(default_factory=list, description="Supporting evidence")
    conversation_id: str = Field(..., description="Conversation identifier")

class ErrorResponse(BaseModel):
    """Standard error response"""
    detail: str = Field(..., description="Error description")
    code: Optional[str] = Field(None, description="Error code")

# Legacy support functions (placeholder implementations)
async def extract_text(file) -> dict:
    """Extract text from uploaded file - placeholder implementation"""
    content = await file.read()
    await file.seek(0)  # Reset file pointer
    
    return {
        "full_text": f"Mock extracted text from {file.filename}",
        "blocks": [
            {
                "text": f"Sample text from {file.filename}",
                "span": {
                    "page": 1,
                    "start_line": 1,
                    "end_line": 1
                }
            }
        ]
    }

def analyze_document(ocr_data: dict, top_k: int = 5) -> dict:
    """Analyze document for clauses and risks"""
    return {
        "clauses": [
            {
                "id": "c-0001",
                "tag": "liability",
                "text": "Sample liability clause",
                "span": {"page": 1, "start_line": 1, "end_line": 1}
            }
        ],
        "risks": [
            {
                "clause_id": "c-0001",
                "level": "medium",
                "score": 0.6,
                "rationale": "Sample financial risk assessment",
                "supporting_context": []
            }
        ],
        "summary_200w": "Document analysis completed with mock data"
    }

def answer_with_vertex(question: str, context: str = "") -> dict:
    """Answer question using RAG with Vertex AI"""
    return {
        "answer": f"Mock answer for: {question}",
        "evidence": [],
        "conversation_id": "mock-conversation-id"
    }
