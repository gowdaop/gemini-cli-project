from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime
import uuid

class ClauseTag(str, Enum):
    """Enhanced clause type classifications"""
    LIABILITY = "liability"
    INDEMNITY = "indemnity" 
    TERMINATION = "termination"
    PAYMENT = "payment"
    IP = "ip"
    CONFIDENTIALITY = "confidentiality"
    GOVERNING_LAW = "governing_law"
    ARBITRATION = "arbitration"
    OTHER = "other"

class RiskLevel(str, Enum):
    """Risk level classifications"""
    WHITE = "white"      # No significant risk
    YELLOW = "yellow"    # Low risk
    ORANGE = "orange"    # Medium risk  
    RED = "red"          # High risk

class ConfidenceLevel(str, Enum):
    """Classification confidence levels"""
    LOW = "low"          # < 0.5
    MEDIUM = "medium"    # 0.5 - 0.8
    HIGH = "high"        # > 0.8

class ClassificationMethod(str, Enum):
    """Classification method used"""
    RULE_BASED = "rule_based"
    ML_BASED = "ml_based"
    HYBRID = "hybrid"
    MANUAL = "manual"

class PageSpan(BaseModel):
    """Represents the location of text within a document"""
    page: int = Field(..., ge=1, description="Page number (1-indexed)")
    start_line: int = Field(..., ge=1, description="Starting line number")
    end_line: int = Field(..., ge=1, description="Ending line number")
    
    @validator('end_line')
    def end_line_must_be_gte_start_line(cls, v, values):
        if 'start_line' in values and v < values['start_line']:
            raise ValueError('end_line must be >= start_line')
        return v

class ClauseMetadata(BaseModel):
    """Metadata for clause classification and analysis"""
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence score")
    classification_method: ClassificationMethod = Field(..., description="Method used for classification")
    matched_keywords: List[str] = Field(default_factory=list, description="Keywords that matched during classification")
    matched_patterns: List[str] = Field(default_factory=list, description="Regex patterns that matched")
    legal_domain: Optional[str] = Field(None, description="Legal domain or area of law")
    language: str = Field(default="en", description="Language of the clause")
    processing_time_ms: Optional[float] = Field(None, description="Time taken to process this clause")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the classification was performed")
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v

class ClauseContext(BaseModel):
    """Additional context for clause understanding"""
    surrounding_text: Optional[str] = Field(None, description="Text surrounding the clause for context")
    document_section: Optional[str] = Field(None, description="Section of document where clause appears")
    clause_number: Optional[str] = Field(None, description="Clause number if available")
    parent_clause_id: Optional[str] = Field(None, description="Parent clause if this is a sub-clause")
    related_clause_ids: List[str] = Field(default_factory=list, description="IDs of related clauses")

class Clause(BaseModel):
    """Enhanced clause model with comprehensive metadata"""
    id: str = Field(..., description="Unique clause identifier")
    tag: ClauseTag = Field(..., description="Type of legal clause")
    text: str = Field(..., min_length=1, description="The clause text")
    span: PageSpan = Field(..., description="Location in document")
    metadata: ClauseMetadata = Field(..., description="Classification metadata")
    context: Optional[ClauseContext] = Field(None, description="Additional context information")
    
    @validator('id')
    def validate_id(cls, v):
        if not v or not v.strip():
            raise ValueError('Clause ID cannot be empty')
        return v.strip()
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Clause text cannot be empty')
        if len(v.strip()) < 10:
            raise ValueError('Clause text must be at least 10 characters')
        return v.strip()
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get confidence level based on score"""
        if self.metadata.confidence < 0.5:
            return ConfidenceLevel.LOW
        elif self.metadata.confidence < 0.8:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.HIGH

class ClauseClassificationRequest(BaseModel):
    """Request model for clause classification"""
    text: str = Field(..., min_length=10, max_length=10000, description="Text to classify")
    context: Optional[str] = Field(None, max_length=5000, description="Additional context for classification")
    document_id: Optional[str] = Field(None, description="Document identifier for tracking")
    language: str = Field(default="en", description="Language of the text")
    classification_method: Optional[ClassificationMethod] = Field(None, description="Preferred classification method")
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text to classify cannot be empty')
        return v.strip()

class ClauseClassificationResponse(BaseModel):
    """Response model for clause classification"""
    clause: Clause = Field(..., description="Classified clause")
    alternatives: List[Dict[str, Any]] = Field(default_factory=list, description="Alternative classifications")
    processing_info: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")

class BulkClauseClassificationRequest(BaseModel):
    """Request model for bulk clause classification"""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to classify")
    document_id: Optional[str] = Field(None, description="Document identifier for tracking")
    language: str = Field(default="en", description="Language of the texts")
    classification_method: Optional[ClassificationMethod] = Field(None, description="Preferred classification method")
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError('Must provide at least one text to classify')
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f'Text at index {i} cannot be empty')
            if len(text.strip()) < 10:
                raise ValueError(f'Text at index {i} must be at least 10 characters')
        return [text.strip() for text in v]

class BulkClauseClassificationResponse(BaseModel):
    """Response model for bulk clause classification"""
    clauses: List[Clause] = Field(..., description="List of classified clauses")
    summary: Dict[str, Any] = Field(..., description="Classification summary statistics")
    processing_info: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")

class ClauseAnalysisRequest(BaseModel):
    """Request model for comprehensive clause analysis"""
    clause: Clause = Field(..., description="Clause to analyze")
    analysis_types: List[str] = Field(default_factory=lambda: ["risk", "compliance", "entities"], 
                                    description="Types of analysis to perform")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional analysis context")

class ClauseAnalysisResponse(BaseModel):
    """Response model for clause analysis"""
    clause_id: str = Field(..., description="ID of analyzed clause")
    risk_analysis: Optional[Dict[str, Any]] = Field(None, description="Risk analysis results")
    compliance_analysis: Optional[Dict[str, Any]] = Field(None, description="Compliance analysis results")
    entity_analysis: Optional[Dict[str, Any]] = Field(None, description="Legal entity analysis results")
    recommendations: List[str] = Field(default_factory=list, description="Analysis-based recommendations")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall analysis confidence")

class ClauseSearchRequest(BaseModel):
    """Request model for clause search"""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    clause_types: Optional[List[ClauseTag]] = Field(None, description="Filter by clause types")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    language: str = Field(default="en", description="Language filter")

class ClauseSearchResponse(BaseModel):
    """Response model for clause search"""
    results: List[Clause] = Field(..., description="Matching clauses")
    total_count: int = Field(..., description="Total number of matches")
    query_metadata: Dict[str, Any] = Field(default_factory=dict, description="Search metadata")

class ClauseStatistics(BaseModel):
    """Statistics about clause classification"""
    total_clauses: int = Field(..., description="Total number of clauses")
    clause_distribution: Dict[ClauseTag, int] = Field(..., description="Distribution by clause type")
    confidence_distribution: Dict[ConfidenceLevel, int] = Field(..., description="Distribution by confidence")
    method_distribution: Dict[ClassificationMethod, int] = Field(..., description="Distribution by method")
    average_confidence: float = Field(..., description="Average confidence score")
    processing_time_stats: Dict[str, float] = Field(default_factory=dict, description="Processing time statistics")

class ClauseValidationError(BaseModel):
    """Model for clause validation errors"""
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Error message")
    value: Optional[Any] = Field(None, description="Invalid value")
    suggestion: Optional[str] = Field(None, description="Suggested fix")

class ClauseValidationResponse(BaseModel):
    """Response model for clause validation"""
    is_valid: bool = Field(..., description="Whether the clause is valid")
    errors: List[ClauseValidationError] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")

# Utility models for API responses
class ClauseServiceStatus(BaseModel):
    """Status model for clause classification service"""
    status: str = Field(..., description="Service status")
    version: str = Field(default="1.0.0", description="Service version")
    features: List[str] = Field(default_factory=list, description="Available features")
    statistics: Optional[ClauseStatistics] = Field(None, description="Service statistics")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update time")

class ClauseExportRequest(BaseModel):
    """Request model for exporting clauses"""
    clause_ids: List[str] = Field(..., min_items=1, description="List of clause IDs to export")
    format: str = Field(default="json", regex="^(json|csv|xml)$", description="Export format")
    include_metadata: bool = Field(default=True, description="Include metadata in export")
    include_context: bool = Field(default=False, description="Include context in export")

class ClauseExportResponse(BaseModel):
    """Response model for clause export"""
    data: Union[str, Dict[str, Any]] = Field(..., description="Exported data")
    format: str = Field(..., description="Export format used")
    clause_count: int = Field(..., description="Number of clauses exported")
    export_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Export timestamp")

# Configuration models
class ClauseClassificationConfig(BaseModel):
    """Configuration for clause classification"""
    default_method: ClassificationMethod = Field(default=ClassificationMethod.HYBRID, description="Default classification method")
    confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum confidence threshold")
    max_text_length: int = Field(default=10000, ge=100, description="Maximum text length for classification")
    enable_caching: bool = Field(default=True, description="Enable result caching")
    cache_ttl_seconds: int = Field(default=3600, ge=60, description="Cache TTL in seconds")
    enable_analytics: bool = Field(default=True, description="Enable analytics collection")
    supported_languages: List[str] = Field(default_factory=lambda: ["en", "es", "fr"], description="Supported languages")

# Create type aliases for common use cases
ClauseList = List[Clause]
ClauseDict = Dict[str, Clause]
ClauseTagDistribution = Dict[ClauseTag, int]

# Utility functions for model validation
def validate_clause_list(clauses: List[Clause]) -> List[ClauseValidationError]:
    """Validate a list of clauses for consistency"""
    errors = []
    clause_ids = set()
    
    for i, clause in enumerate(clauses):
        # Check for duplicate IDs
        if clause.id in clause_ids:
            errors.append(ClauseValidationError(
                field=f"clauses[{i}].id",
                message="Duplicate clause ID found",
                value=clause.id,
                suggestion="Ensure all clause IDs are unique"
            ))
        clause_ids.add(clause.id)
        
        # Check for overlapping spans on same page
        for j, other_clause in enumerate(clauses[i+1:], i+1):
            if (clause.span.page == other_clause.span.page and
                not (clause.span.end_line < other_clause.span.start_line or 
                     other_clause.span.end_line < clause.span.start_line)):
                errors.append(ClauseValidationError(
                    field=f"clauses[{i}].span",
                    message=f"Overlapping spans with clause {j}",
                    value=None,
                    suggestion="Check clause boundaries for accuracy"
                ))
    
    return errors

def create_clause_id(document_id: Optional[str] = None, index: int = 0) -> str:
    """Generate a unique clause ID"""
    if document_id:
        return f"{document_id}-c-{index+1:04d}"
    else:
        return f"c-{uuid.uuid4().hex[:8]}-{index+1:04d}"

def clause_to_dict(clause: Clause, include_metadata: bool = True, include_context: bool = False) -> Dict[str, Any]:
    """Convert clause to dictionary with optional fields"""
    result = {
        "id": clause.id,
        "tag": clause.tag.value,
        "text": clause.text,
        "span": clause.span.dict()
    }
    
    if include_metadata:
        result["metadata"] = clause.metadata.dict()
        result["confidence_level"] = clause.confidence_level.value
    
    if include_context and clause.context:
        result["context"] = clause.context.dict()
    
    return result

# Export all models for easy importing
__all__ = [
    "ClauseTag", "RiskLevel", "ConfidenceLevel", "ClassificationMethod",
    "PageSpan", "ClauseMetadata", "ClauseContext", "Clause",
    "ClauseClassificationRequest", "ClauseClassificationResponse",
    "BulkClauseClassificationRequest", "BulkClauseClassificationResponse",
    "ClauseAnalysisRequest", "ClauseAnalysisResponse",
    "ClauseSearchRequest", "ClauseSearchResponse",
    "ClauseStatistics", "ClauseValidationError", "ClauseValidationResponse",
    "ClauseServiceStatus", "ClauseExportRequest", "ClauseExportResponse",
    "ClauseClassificationConfig",
    "ClauseList", "ClauseDict", "ClauseTagDistribution",
    "validate_clause_list", "create_clause_id", "clause_to_dict"
]
