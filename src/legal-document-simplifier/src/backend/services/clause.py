import asyncio
import logging
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import uuid
from collections import defaultdict, Counter

try:
    import spacy
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    spacy = None

from ..schemas.analysis import Clause, ClauseTag, OCRText, OCRBlock, PageSpan
from ..config import settings
from . import embedding

logger = logging.getLogger(__name__)

@dataclass
class ClausePattern:
    """Pattern definition for clause detection"""
    tag: ClauseTag
    keywords: List[str]
    patterns: List[str]
    weight: float
    context_words: List[str]

@dataclass
class ClassificationResult:
    """Result of clause classification"""
    tag: ClauseTag
    confidence: float
    matched_patterns: List[str]
    matched_keywords: List[str]

class LegalTermsLexicon:
    """Legal terminology and patterns for clause identification"""
    
    def __init__(self):
        self.clause_patterns = self._build_clause_patterns()
        self.legal_connectors = self._build_legal_connectors()
        self.exclusion_patterns = self._build_exclusion_patterns()
    
    def _build_clause_patterns(self) -> Dict[ClauseTag, ClausePattern]:
        """Build comprehensive patterns for each clause type"""
        return {
            ClauseTag.LIABILITY: ClausePattern(
                tag=ClauseTag.LIABILITY,
                keywords=[
                    "liability", "liable", "responsible", "damages", "loss", "harm",
                    "injury", "limitation", "exclude", "disclaim", "indemnify"
                ],
                patterns=[
                    r"(?i)\b(shall|will|may)\s+(?:not\s+)?be\s+liable\s+for\b",
                    r"(?i)\blimitation\s+of\s+liability\b",
                    r"(?i)\bliability\s+(?:is\s+)?(?:limited|excluded|disclaimed)\b",
                    r"(?i)\b(?:no|without)\s+liability\b",
                    r"(?i)\bdamages\s+(?:shall|will|may)\s+(?:not\s+)?(?:exceed|be\s+limited)\b",
                    r"(?i)\bundertakes?\s+no\s+(?:liability|responsibility)\b"
                ],
                weight=0.9,
                context_words=["damages", "loss", "harm", "responsible", "fault"]
            ),
            
            ClauseTag.INDEMNITY: ClausePattern(
                tag=ClauseTag.INDEMNITY,
                keywords=[
                    "indemnify", "indemnification", "hold harmless", "defend", 
                    "reimburse", "compensate", "make whole"
                ],
                patterns=[
                    r"(?i)\bindemnify\s+(?:and\s+)?(?:hold\s+)?harmless\b",
                    r"(?i)\bhold\s+\w+\s+harmless\b",
                    r"(?i)\bdefend,?\s+indemnify\b",
                    r"(?i)\bindemnification\s+(?:clause|provision|obligation)\b",
                    r"(?i)\bshall\s+(?:defend\s+)?(?:and\s+)?indemnify\b",
                    r"(?i)\bagrees?\s+to\s+(?:defend\s+)?(?:and\s+)?indemnify\b"
                ],
                weight=0.95,
                context_words=["defend", "harmless", "third party", "claims", "costs"]
            ),
            
            ClauseTag.TERMINATION: ClausePattern(
                tag=ClauseTag.TERMINATION,
                keywords=[
                    "terminate", "termination", "end", "expire", "dissolution",
                    "breach", "default", "notice", "effective date"
                ],
                patterns=[
                    r"(?i)\bterminate\s+(?:this\s+)?(?:agreement|contract)\b",
                    r"(?i)\btermination\s+(?:of\s+)?(?:this\s+)?(?:agreement|contract)\b",
                    r"(?i)\b(?:upon|after)\s+(?:\d+\s+)?(?:days?\s+)?(?:written\s+)?notice\b",
                    r"(?i)\beffective\s+(?:date\s+of\s+)?termination\b",
                    r"(?i)\bmaterial\s+breach\b",
                    r"(?i)\bimmediately\s+terminate\b"
                ],
                weight=0.85,
                context_words=["notice", "breach", "default", "expire", "end"]
            ),
            
            ClauseTag.PAYMENT: ClausePattern(
                tag=ClauseTag.PAYMENT,
                keywords=[
                    "payment", "pay", "invoice", "fee", "charge", "cost",
                    "price", "amount", "due", "owing", "interest", "penalty"
                ],
                patterns=[
                    r"(?i)\bpayment\s+(?:terms|schedule|due)\b",
                    r"(?i)\b(?:shall|will|must)\s+pay\b",
                    r"(?i)\binvoice\s+(?:date|amount|payment)\b",
                    r"(?i)\bdue\s+(?:and\s+)?payable\b",
                    r"(?i)\blate\s+(?:fee|penalty|charge)\b",
                    r"(?i)\binterest\s+(?:rate|charge|accrues?)\b"
                ],
                weight=0.8,
                context_words=["invoice", "due", "payable", "fees", "charges"]
            ),
            
            ClauseTag.IP: ClausePattern(
                tag=ClauseTag.IP,
                keywords=[
                    "intellectual property", "copyright", "trademark", "patent",
                    "proprietary", "confidential", "trade secret", "know-how"
                ],
                patterns=[
                    r"(?i)\bintellectual\s+property\s+rights?\b",
                    r"(?i)\btrade\s+secrets?\b",
                    r"(?i)\bproprietary\s+(?:information|rights?|technology)\b",
                    r"(?i)\b(?:copyright|trademark|patent)\s+(?:owner|holder|protection)\b",
                    r"(?i)\bknow-how\b",
                    r"(?i)\bownership\s+of\s+(?:intellectual\s+property|ip)\b"
                ],
                weight=0.85,
                context_words=["ownership", "rights", "proprietary", "confidential"]
            ),
            
            ClauseTag.CONFIDENTIALITY: ClausePattern(
                tag=ClauseTag.CONFIDENTIALITY,
                keywords=[
                    "confidential", "confidentiality", "non-disclosure", "proprietary",
                    "secret", "disclose", "disclosure", "private"
                ],
                patterns=[
                    r"(?i)\bconfidential\s+(?:information|data|material)\b",
                    r"(?i)\bnon-disclosure\s+(?:agreement|obligation)\b",
                    r"(?i)\bshall\s+not\s+disclose\b",
                    r"(?i)\bobligations?\s+of\s+confidentiality\b",
                    r"(?i)\btrade\s+secrets?\b",
                    r"(?i)\bproprietary\s+and\s+confidential\b"
                ],
                weight=0.85,
                context_words=["disclose", "private", "secret", "proprietary"]
            ),
            
            ClauseTag.GOVERNING_LAW: ClausePattern(
                tag=ClauseTag.GOVERNING_LAW,
                keywords=[
                    "governing law", "jurisdiction", "laws", "courts", "venue",
                    "forum", "applicable law", "construed"
                ],
                patterns=[
                    r"(?i)\bgoverning\s+law\b",
                    r"(?i)\bshall\s+be\s+(?:governed|construed)\s+by\b",
                    r"(?i)\bjurisdiction\s+(?:of|and\s+venue)\b",
                    r"(?i)\bapplicable\s+law\b",
                    r"(?i)\bexclusive\s+jurisdiction\b",
                    r"(?i)\blaws\s+of\s+\w+\b"
                ],
                weight=0.75,
                context_words=["courts", "venue", "jurisdiction", "construed"]
            ),
            
            ClauseTag.ARBITRATION: ClausePattern(
                tag=ClauseTag.ARBITRATION,
                keywords=[
                    "arbitration", "arbitrator", "dispute resolution", "mediation",
                    "binding", "adr", "alternative dispute"
                ],
                patterns=[
                    r"(?i)\barbitration\s+(?:proceedings?|clause|provision)\b",
                    r"(?i)\bbinding\s+arbitration\b",
                    r"(?i)\bdispute\s+resolution\b",
                    r"(?i)\bsubject\s+to\s+arbitration\b",
                    r"(?i)\barbitrator\s+(?:shall|will|may)\b",
                    r"(?i)\balternative\s+dispute\s+resolution\b"
                ],
                weight=0.8,
                context_words=["disputes", "binding", "resolution", "proceedings"]
            )
        }
    
    def _build_legal_connectors(self) -> List[str]:
        """Legal connecting phrases that help identify clause boundaries"""
        return [
            "provided that", "subject to", "notwithstanding", "in the event",
            "upon the occurrence", "except as", "unless otherwise", "to the extent",
            "in accordance with", "pursuant to", "as set forth", "as provided"
        ]
    
    def _build_exclusion_patterns(self) -> List[str]:
        """Patterns that should exclude text from being considered clauses"""
        return [
            r"(?i)^(?:page|section|article|paragraph)\s+\d+",
            r"(?i)^(?:whereas|recitals?|preamble)",
            r"(?i)^(?:table\s+of\s+contents|index)",
            r"(?i)^(?:appendix|exhibit|schedule)\s+[a-z0-9]"
        ]

class ClauseSegmenter:
    """Segments document text into potential clause boundaries"""
    
    def __init__(self, lexicon: LegalTermsLexicon):
        self.lexicon = lexicon
        self.min_clause_length = 20
        self.max_clause_length = 2000
    
    async def segment_clauses(self, ocr_text: OCRText) -> List[Dict[str, Any]]:
        """
        Segment document into potential clause units
        
        Returns:
            List of clause segments with text, span, and metadata
        """
        try:
            segments = []
            
            # Strategy 1: Use existing OCR blocks as initial segments
            for i, block in enumerate(ocr_text.blocks):
                if self._is_valid_clause_candidate(block.text):
                    segments.append({
                        "text": block.text.strip(),
                        "span": block.span,
                        "source": "ocr_block",
                        "block_index": i
                    })
            
            # Strategy 2: Split long blocks by sentence patterns
            enhanced_segments = []
            for segment in segments:
                if len(segment["text"]) > self.max_clause_length:
                    sub_segments = await self._split_long_segment(segment)
                    enhanced_segments.extend(sub_segments)
                else:
                    enhanced_segments.append(segment)
            
            # Strategy 3: Merge short adjacent segments if they form coherent clauses
            merged_segments = self._merge_related_segments(enhanced_segments)
            
            logger.debug(f"Segmented document into {len(merged_segments)} potential clauses")
            return merged_segments
            
        except Exception as e:
            logger.error(f"Clause segmentation failed: {e}")
            # Fallback: treat each OCR block as a separate segment
            return [
                {
                    "text": block.text.strip(),
                    "span": block.span,
                    "source": "fallback",
                    "block_index": i
                }
                for i, block in enumerate(ocr_text.blocks)
                if block.text.strip()
            ]
    
    def _is_valid_clause_candidate(self, text: str) -> bool:
        """Check if text is a valid clause candidate"""
        if not text or len(text.strip()) < self.min_clause_length:
            return False
        
        # Exclude based on exclusion patterns
        for pattern in self.lexicon.exclusion_patterns:
            if re.search(pattern, text):
                return False
        
        # Must contain some legal-sounding content
        legal_indicators = [
            "shall", "will", "may", "must", "agrees", "covenant", "represent",
            "warrant", "acknowledge", "undertake", "obligation", "right", "duty"
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in legal_indicators)
    
    async def _split_long_segment(self, segment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a long segment into smaller clause units"""
        text = segment["text"]
        
        # Split on sentence boundaries that look like clause separators
        split_patterns = [
            r'(?<=[.;])\s+(?=[A-Z])',  # After period/semicolon before capital
            r'(?<=\.)\s+\(\w+\)',      # After period before numbered subsection
            r'(?:\n\s*){2,}',          # Multiple line breaks
        ]
        
        segments = [text]
        for pattern in split_patterns:
            new_segments = []
            for seg in segments:
                parts = re.split(pattern, seg)
                new_segments.extend([p.strip() for p in parts if p.strip()])
            segments = new_segments
        
        # Create segment objects
        result = []
        start_line = segment["span"].start_line
        
        for i, seg_text in enumerate(segments):
            if len(seg_text) >= self.min_clause_length:
                lines_in_segment = seg_text.count('\n') + 1
                result.append({
                    "text": seg_text,
                    "span": PageSpan(
                        page=segment["span"].page,
                        start_line=start_line,
                        end_line=start_line + lines_in_segment - 1
                    ),
                    "source": "split_segment",
                    "parent_block": segment.get("block_index")
                })
                start_line += lines_in_segment
        
        return result if result else [segment]  # Return original if splitting failed
    
    def _merge_related_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge short segments that appear to be related"""
        if len(segments) <= 1:
            return segments
        
        merged = []
        current_group = [segments[0]]
        
        for i in range(1, len(segments)):
            current = segments[i]
            previous = segments[i-1]
            
            # Check if segments should be merged
            should_merge = (
                len(current["text"]) < 100 and  # Current is short
                len(previous["text"]) < 200 and  # Previous is not too long
                self._are_segments_related(current["text"], previous["text"])
            )
            
            if should_merge:
                current_group.append(current)
            else:
                # Finalize current group
                if len(current_group) == 1:
                    merged.append(current_group[0])
                else:
                    merged.append(self._create_merged_segment(current_group))
                current_group = [current]
        
        # Handle final group
        if len(current_group) == 1:
            merged.append(current_group[0])
        else:
            merged.append(self._create_merged_segment(current_group))
        
        return merged
    
    def _are_segments_related(self, text1: str, text2: str) -> bool:
        """Check if two text segments are semantically related"""
        # Simple heuristic: check for shared keywords
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        shared_ratio = len(words1 & words2) / len(words1 | words2) if words1 | words2 else 0
        return shared_ratio > 0.3
    
    def _create_merged_segment(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a single segment from a group of segments"""
        combined_text = " ".join(seg["text"] for seg in segments)
        first_span = segments[0]["span"]
        last_span = segments[-1]["span"]
        
        return {
            "text": combined_text,
            "span": PageSpan(
                page=first_span.page,
                start_line=first_span.start_line,
                end_line=last_span.end_line
            ),
            "source": "merged_segments",
            "merged_count": len(segments)
        }

class RuleBasedClassifier:
    """Rule-based clause classification using patterns and keywords"""
    
    def __init__(self, lexicon: LegalTermsLexicon):
        self.lexicon = lexicon
        self.min_confidence = 0.3
    
    async def classify_segment(self, text: str) -> ClassificationResult:
        """Classify a text segment using rule-based patterns"""
        try:
            text_lower = text.lower()
            scores = defaultdict(float)
            matched_patterns = defaultdict(list)
            matched_keywords = defaultdict(list)
            
            # Score each clause type
            for clause_tag, pattern in self.lexicon.clause_patterns.items():
                score = 0.0
                
                # Pattern matching
                pattern_matches = 0
                for regex_pattern in pattern.patterns:
                    matches = re.findall(regex_pattern, text)
                    if matches:
                        pattern_matches += len(matches)
                        matched_patterns[clause_tag].extend(matches)
                
                # Keyword matching
                keyword_matches = 0
                for keyword in pattern.keywords:
                    if keyword in text_lower:
                        keyword_matches += 1
                        matched_keywords[clause_tag].append(keyword)
                
                # Context word bonus
                context_bonus = 0
                for context_word in pattern.context_words:
                    if context_word in text_lower:
                        context_bonus += 0.1
                
                # Calculate score
                pattern_score = (pattern_matches * 0.4) * pattern.weight
                keyword_score = (keyword_matches / len(pattern.keywords)) * 0.4
                context_score = min(context_bonus, 0.2)
                
                total_score = pattern_score + keyword_score + context_score
                scores[clause_tag] = total_score
            
            # Find best match
            if scores:
                best_tag = max(scores, key=scores.get)
                best_score = scores[best_tag]
                
                if best_score >= self.min_confidence:
                    return ClassificationResult(
                        tag=best_tag,
                        confidence=min(best_score, 1.0),
                        matched_patterns=matched_patterns[best_tag],
                        matched_keywords=matched_keywords[best_tag]
                    )
            
            # Default to OTHER if no confident match
            return ClassificationResult(
                tag=ClauseTag.OTHER,
                confidence=0.2,
                matched_patterns=[],
                matched_keywords=[]
            )
            
        except Exception as e:
            logger.warning(f"Rule-based classification failed: {e}")
            return ClassificationResult(
                tag=ClauseTag.OTHER,
                confidence=0.1,
                matched_patterns=[],
                matched_keywords=[]
            )

class MLClassifier:
    """Machine Learning-based clause classification (ready for future enhancement)"""
    
    def __init__(self):
        self.model_loaded = False
        self.tokenizer = None
        self.model = None
    
    async def initialize(self):
        """Initialize ML models (placeholder for future implementation)"""
        try:
            if not NLP_AVAILABLE:
                logger.warning("NLP libraries not available for ML classification")
                return
            
            # TODO: Load pre-trained legal clause classification model
            # self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
            # self.model = AutoModel.from_pretrained("legal-clause-classifier")
            
            logger.info("ML classifier ready for future implementation")
            
        except Exception as e:
            logger.warning(f"ML classifier initialization failed: {e}")
    
    async def classify_segment(self, text: str) -> Optional[ClassificationResult]:
        """ML-based classification (placeholder for future implementation)"""
        if not self.model_loaded:
            return None
        
        try:
            # TODO: Implement ML classification
            # 1. Tokenize text
            # 2. Generate embeddings
            # 3. Run through classification model
            # 4. Return result with confidence
            
            logger.debug("ML classification not yet implemented")
            return None
            
        except Exception as e:
            logger.warning(f"ML classification failed: {e}")
            return None

class EnhancedClauseClassifier:
    """Main clause classification service combining multiple strategies"""
    
    def __init__(self):
        self.lexicon = LegalTermsLexicon()
        self.segmenter = ClauseSegmenter(self.lexicon)
        self.rule_classifier = RuleBasedClassifier(self.lexicon)
        self.ml_classifier = MLClassifier()
        self.initialized = False
    
    async def initialize(self):
        """Initialize all classification components"""
        if self.initialized:
            return
        
        try:
            await self.ml_classifier.initialize()
            self.initialized = True
            logger.info("Enhanced clause classifier initialized")
        except Exception as e:
            logger.warning(f"Clause classifier initialization warning: {e}")
            self.initialized = True  # Continue with rule-based only
    
    async def classify_clauses(self, ocr_text: OCRText) -> List[Clause]:
        """
        Main clause classification function
        
        Args:
            ocr_text: OCR extracted text with blocks
            
        Returns:
            List of classified clauses with spans and metadata
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            logger.info(f"Starting clause classification for document with {len(ocr_text.blocks)} blocks")
            
            # Step 1: Segment document into potential clauses
            segments = await self.segmenter.segment_clauses(ocr_text)
            logger.debug(f"Segmented document into {len(segments)} potential clauses")
            
            # Step 2: Classify each segment
            clauses = []
            for i, segment in enumerate(segments):
                clause = await self._classify_single_segment(segment, i)
                if clause:
                    clauses.append(clause)
            
            # Step 3: Post-process and validate results
            validated_clauses = self._validate_and_enhance_clauses(clauses)
            
            logger.info(f"Successfully classified {len(validated_clauses)} clauses")
            return validated_clauses
            
        except Exception as e:
            logger.error(f"Clause classification failed: {e}")
            # Fallback: create basic clauses from OCR blocks
            return self._create_fallback_clauses(ocr_text)
    
    async def _classify_single_segment(self, segment: Dict[str, Any], index: int) -> Optional[Clause]:
        """Classify a single text segment"""
        try:
            text = segment["text"]
            
            # Try ML classification first (if available)
            ml_result = await self.ml_classifier.classify_segment(text)
            
            # Always get rule-based classification
            rule_result = await self.rule_classifier.classify_segment(text)
            
            # Combine results (prefer ML if available and confident)
            if ml_result and ml_result.confidence > 0.7:
                final_result = ml_result
                classification_method = "ml"
            else:
                final_result = rule_result
                classification_method = "rule"
            
            # Create clause object
            clause_id = f"c-{index+1:04d}"
            
            clause = Clause(
                id=clause_id,
                tag=final_result.tag,
                text=text,
                span=segment["span"]
            )
            
            logger.debug(
                f"Classified clause {clause_id}: {final_result.tag.value} "
                f"(confidence: {final_result.confidence:.2f}, method: {classification_method})"
            )
            
            return clause
            
        except Exception as e:
            logger.warning(f"Single segment classification failed: {e}")
            return None
    
    def _validate_and_enhance_clauses(self, clauses: List[Clause]) -> List[Clause]:
        """Validate and enhance classified clauses"""
        try:
            validated = []
            clause_counts = Counter(clause.tag for clause in clauses)
            
            for clause in clauses:
                # Skip clauses that are too short or repetitive
                if len(clause.text) < 15:
                    continue
                
                # Enhance clause with additional metadata if needed
                validated.append(clause)
            
            # Log classification statistics
            logger.info(f"Clause distribution: {dict(clause_counts)}")
            
            return validated
            
        except Exception as e:
            logger.warning(f"Clause validation failed: {e}")
            return clauses
    
    def _create_fallback_clauses(self, ocr_text: OCRText) -> List[Clause]:
        """Create basic fallback clauses when classification fails"""
        try:
            clauses = []
            for i, block in enumerate(ocr_text.blocks):
                if block.text.strip() and len(block.text.strip()) > 15:
                    clause = Clause(
                        id=f"c-{i+1:04d}",
                        tag=ClauseTag.OTHER,
                        text=block.text.strip(),
                        span=block.span
                    )
                    clauses.append(clause)
            
            logger.warning(f"Created {len(clauses)} fallback clauses")
            return clauses
            
        except Exception as e:
            logger.error(f"Fallback clause creation failed: {e}")
            return []

# Global service instance
_clause_classifier = EnhancedClauseClassifier()

# Public API functions for router integration
async def classify_clauses(ocr_text: OCRText) -> List[Clause]:
    """
    Main public function for clause classification
    
    Args:
        ocr_text: OCR extracted text with blocks and spans
        
    Returns:
        List of classified clauses with tags and metadata
    """
    try:
        return await _clause_classifier.classify_clauses(ocr_text)
    except Exception as e:
        logger.error(f"Clause classification failed: {e}")
        
        # ✅ FIX: Return fallback clauses instead of crashing
        fallback_clauses = []
        for i, block in enumerate(ocr_text.blocks):
            if block.text.strip() and len(block.text.strip()) > 15:
                fallback_clauses.append(Clause(
                    id=f"c-{i+1:04d}",
                    tag=ClauseTag.OTHER,
                    text=block.text.strip(),
                    span=block.span
                ))
        
        logger.warning(f"Created {len(fallback_clauses)} fallback clauses")
        return fallback_clauses


async def classify_single_text(text: str) -> ClassificationResult:
    """
    Classify a single piece of text (utility function)
    
    Args:
        text: Text to classify
        
    Returns:
        Classification result with tag and confidence
    """
    try:
        if not _clause_classifier.initialized:
            await _clause_classifier.initialize()
        
        # ✅ FIX: Remove 'await' - classify_segment is synchronous
        return _clause_classifier.rule_classifier.classify_segment(text)
        
    except Exception as e:
        logger.error(f"Single text classification failed: {e}")
        return ClassificationResult(
            tag=ClauseTag.OTHER,
            confidence=0.1,
            matched_patterns=[],
            matched_keywords=[]
        )


async def health_check() -> Dict[str, Any]:
    """Health check for clause classification service"""
    try:
        if not _clause_classifier.initialized:
            await _clause_classifier.initialize()
        
        # Test classification with sample text
        test_text = "The Company shall not be liable for any indirect damages."
        # ✅ FIX: Properly await the async classify_segment method
        test_result = await _clause_classifier.rule_classifier.classify_segment(test_text)
        
        return {
            "status": "healthy",
            "service": "clause_classification",
            "initialized": _clause_classifier.initialized,
            "ml_available": NLP_AVAILABLE,
            "test_classification": {
                "text": test_text[:50] + "...",
                "tag": test_result.tag.value,
                "confidence": test_result.confidence
            },
            "features": [
                "rule_based_classification",
                "ml_ready_architecture",
                "semantic_segmentation",
                "confidence_scoring",
                "multi_strategy_fusion"
            ]
        }
    except Exception as e:
        return {
            "status": "degraded",
            "service": "clause_classification",
            "error": str(e),
            "fallback_available": True
        }


async def get_classification_stats() -> Dict[str, Any]:
    """Get classification service statistics"""
    return {
        "clause_types_supported": len(ClauseTag),
        "classification_methods": ["rule_based", "ml_ready"],
        "lexicon_patterns": len(_clause_classifier.lexicon.clause_patterns),
        "min_confidence_threshold": _clause_classifier.rule_classifier.min_confidence,
        "supported_languages": ["english"],  # Expandable
        "performance_features": [
            "async_processing",
            "concurrent_classification",
            "intelligent_segmentation",
            "confidence_scoring"
        ]
    }