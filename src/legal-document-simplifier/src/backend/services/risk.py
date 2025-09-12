import asyncio
import logging
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict 
import json

from ..schemas.analysis import (
    Clause, 
    ClauseTag, 
    RiskScore, 
    RiskLevel, 
    RAGContextItem
)
from ..config import settings
from . import rag

logger = logging.getLogger(__name__)

@dataclass
class RiskAssessmentConfig:
    """Configuration for risk assessment parameters"""
    base_risk_weights: Dict[ClauseTag, float]
    risk_thresholds: Dict[str, float]
    evidence_weight: float
    similarity_threshold: float
    max_evidence_items: int

class RiskCalculator:
    """Advanced risk calculation engine with RAG enhancement"""
    
    def __init__(self):
        self.config = self._load_risk_config()
        self._risk_cache = {}
        self._cache_ttl = 3600  # 1 hour
    
    def _load_risk_config(self) -> RiskAssessmentConfig:
        """Load risk assessment configuration"""
        return RiskAssessmentConfig(
            base_risk_weights={
                ClauseTag.LIABILITY: 0.85,           # High risk - liability exposure
                ClauseTag.INDEMNITY: 0.80,           # High risk - financial exposure
                ClauseTag.TERMINATION: 0.65,         # Medium-high risk - business continuity
                ClauseTag.PAYMENT: 0.60,             # Medium-high risk - cash flow
                ClauseTag.IP: 0.55,                  # Medium risk - intellectual property
                ClauseTag.CONFIDENTIALITY: 0.45,    # Medium-low risk - information
                ClauseTag.GOVERNING_LAW: 0.35,       # Low risk - jurisdiction
                ClauseTag.ARBITRATION: 0.30,         # Low risk - dispute resolution
                ClauseTag.OTHER: 0.20                # Lowest risk - miscellaneous
            },
            risk_thresholds={
                "red": 0.75,      # High risk threshold
                "orange": 0.50,   # Medium risk threshold  
                "yellow": 0.25,   # Low risk threshold
                "white": 0.0      # No significant risk
            },
            evidence_weight=0.30,        # 30% weight for RAG evidence
            similarity_threshold=0.70,   # Minimum similarity for evidence consideration
            max_evidence_items=8        # Maximum evidence items to analyze
        )
    
    async def calculate_clause_risk(
        self, 
        clause: Clause, 
        evidence: Optional[List[RAGContextItem]] = None
    ) -> Tuple[float, RiskLevel, str]:
        """
        Calculate comprehensive risk score for a clause with RAG evidence
        
        Returns:
            Tuple of (risk_score, risk_level, rationale)
        """
        try:
            # Get base risk from clause type
            base_risk = self.config.base_risk_weights.get(clause.tag, 0.2)
            
            # Retrieve evidence if not provided
            if evidence is None:
                evidence = await self._retrieve_clause_evidence(clause)
            
            # Calculate evidence-adjusted risk
            evidence_adjustment = self._calculate_evidence_adjustment(clause, evidence)
            
            # Calculate final risk score
            final_risk = self._combine_risk_scores(base_risk, evidence_adjustment)
            
            # Determine risk level
            risk_level = self._determine_risk_level(final_risk)
            
            # Generate rationale
            rationale = self._generate_risk_rationale(
                clause, final_risk, risk_level, evidence
            )
            
            logger.debug(
                f"Risk calculated for clause {clause.id}: "
                f"base={base_risk:.2f}, evidence_adj={evidence_adjustment:.2f}, "
                f"final={final_risk:.2f}, level={risk_level.value}"
            )
            
            return final_risk, risk_level, rationale
            
        except Exception as e:
            logger.error(f"Risk calculation failed for clause {clause.id}: {e}")
            # Return conservative high-risk assessment as fallback
            return 0.75, RiskLevel.ORANGE, self._create_fallback_rationale(clause)
    
    async def _retrieve_clause_evidence(self, clause: Clause) -> List[RAGContextItem]:
        """Retrieve relevant evidence for clause risk assessment"""
        try:
            # Create risk-focused query for the clause
            risk_query = self._create_risk_query(clause)
            
            # Retrieve evidence using RAG service
            evidence = await rag.retrieve_contexts(
                query=risk_query,
                top_k=self.config.max_evidence_items
            )
            
            # Filter evidence by similarity threshold
            filtered_evidence = [
                item for item in evidence 
                if item.similarity >= self.config.similarity_threshold
            ]
            
            logger.debug(f"Retrieved {len(filtered_evidence)} relevant evidence items for clause {clause.id}")
            return filtered_evidence
            
        except Exception as e:
            logger.warning(f"Evidence retrieval failed for clause {clause.id}: {e}")
            return []
    
    def _create_risk_query(self, clause: Clause) -> str:
        """Create optimized query for retrieving risk-relevant evidence"""
        clause_type = clause.tag.value.replace("_", " ")
        
        risk_keywords = {
            ClauseTag.LIABILITY: "liability limitation damages unlimited exposure risk",
            ClauseTag.INDEMNITY: "indemnification hold harmless defend damages costs",
            ClauseTag.TERMINATION: "termination notice period breach consequences",
            ClauseTag.PAYMENT: "payment terms late fees interest penalties default",
            ClauseTag.IP: "intellectual property ownership rights infringement",
            ClauseTag.CONFIDENTIALITY: "confidentiality disclosure obligations breach",
            ClauseTag.GOVERNING_LAW: "governing law jurisdiction choice forum",
            ClauseTag.ARBITRATION: "arbitration dispute resolution mediation binding",
            ClauseTag.OTHER: f"{clause_type} clause terms conditions"
        }
        
        keywords = risk_keywords.get(clause.tag, clause_type)
        
        # Combine clause text with risk-specific keywords
        risk_query = f"{clause.text[:200]} {keywords} risk assessment legal precedent"
        
        return risk_query.strip()
    
    def _calculate_evidence_adjustment(
        self, 
        clause: Clause, 
        evidence: List[RAGContextItem]
    ) -> float:
        """Calculate risk adjustment based on RAG evidence analysis"""
        if not evidence:
            return 0.0
        
        try:
            risk_indicators = self._analyze_risk_indicators(evidence)
            protective_factors = self._analyze_protective_factors(evidence)
            precedent_severity = self._analyze_precedent_severity(evidence)
            
            # Calculate weighted evidence score
            evidence_score = (
                risk_indicators * 0.4 +
                precedent_severity * 0.4 +
                protective_factors * 0.2
            )
            
            # Apply evidence weight to get final adjustment
            adjustment = evidence_score * self.config.evidence_weight
            
            logger.debug(
                f"Evidence analysis for clause {clause.id}: "
                f"risk_indicators={risk_indicators:.2f}, "
                f"protective_factors={protective_factors:.2f}, "
                f"precedent_severity={precedent_severity:.2f}, "
                f"adjustment={adjustment:.2f}"
            )
            
            return adjustment
            
        except Exception as e:
            logger.warning(f"Evidence analysis failed: {e}")
            return 0.0
    
    def _analyze_risk_indicators(self, evidence: List[RAGContextItem]) -> float:
        """Analyze evidence for risk-increasing indicators"""
        high_risk_terms = [
            "unlimited", "without limitation", "no limit", "severe penalty",
            "immediate termination", "substantial damages", "criminal liability",
            "personal liability", "joint and several", "indemnify fully"
        ]
        
        medium_risk_terms = [
            "liquidated damages", "material breach", "injunctive relief",
            "specific performance", "attorney fees", "court costs"
        ]
        
        risk_score = 0.0
        total_content = ""
        
        for item in evidence:
            total_content += item.content.lower() + " "
        
        # Count high-risk indicators
        high_risk_count = sum(
            1 for term in high_risk_terms 
            if term in total_content
        )
        
        # Count medium-risk indicators  
        medium_risk_count = sum(
            1 for term in medium_risk_terms
            if term in total_content
        )
        
        # Calculate risk score (0.0 to 1.0)
        risk_score = min(
            (high_risk_count * 0.3 + medium_risk_count * 0.15),
            1.0
        )
        
        return risk_score
    
    def _analyze_protective_factors(self, evidence: List[RAGContextItem]) -> float:
        """Analyze evidence for risk-reducing protective factors"""
        protective_terms = [
            "limited to", "capped at", "reasonable", "good faith",
            "commercially reasonable", "industry standard", "best efforts",
            "mutual", "reciprocal", "proportional", "mitigation"
        ]
        
        protection_score = 0.0
        total_content = ""
        
        for item in evidence:
            total_content += item.content.lower() + " "
        
        # Count protective indicators
        protection_count = sum(
            1 for term in protective_terms
            if term in total_content
        )
        
        # Calculate protection score (0.0 to 1.0, inverse relationship)
        protection_score = min(protection_count * 0.1, 0.5)
        
        # Return negative adjustment (reduces risk)
        return -protection_score
    
    def _analyze_precedent_severity(self, evidence: List[RAGContextItem]) -> float:
        """Analyze legal precedents for severity indicators"""
        severity_indicators = [
            "court found", "held liable", "damages awarded", "breach resulted",
            "penalty imposed", "violation", "default judgment", "adverse ruling"
        ]
        
        severity_score = 0.0
        
        for item in evidence:
            content_lower = item.content.lower()
            
            # Weight by similarity to current clause
            similarity_weight = item.similarity
            
            # Count severity indicators
            severity_count = sum(
                1 for indicator in severity_indicators
                if indicator in content_lower
            )
            
            # Add weighted severity contribution
            severity_score += (severity_count * 0.2 * similarity_weight)
        
        return min(severity_score, 1.0)
    
    def _combine_risk_scores(self, base_risk: float, evidence_adjustment: float) -> float:
        """Combine base risk with evidence-based adjustment"""
        # Ensure base risk is within bounds
        base_risk = max(0.0, min(1.0, base_risk))
        
        # Apply evidence adjustment
        combined_risk = base_risk + evidence_adjustment
        
        # Ensure final risk is within bounds
        final_risk = max(0.0, min(1.0, combined_risk))
        
        return final_risk
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level based on configured thresholds"""
        if risk_score >= self.config.risk_thresholds["red"]:
            return RiskLevel.RED
        elif risk_score >= self.config.risk_thresholds["orange"]:
            return RiskLevel.ORANGE
        elif risk_score >= self.config.risk_thresholds["yellow"]:
            return RiskLevel.YELLOW
        else:
            return RiskLevel.WHITE
    
    def _generate_risk_rationale(
        self,
        clause: Clause,
        risk_score: float,
        risk_level: RiskLevel,
        evidence: List[RAGContextItem]
    ) -> str:
        """Generate human-readable risk assessment rationale"""
        try:
            clause_type = clause.tag.value.replace("_", " ").title()
            evidence_count = len(evidence)
            
            # Base rationale
            rationale_parts = [
                f"This {clause_type} clause has been assessed with a {risk_score:.1f} risk score."
            ]
            
            # Risk level explanation
            risk_explanations = {
                RiskLevel.RED: "This represents a HIGH RISK that requires immediate legal review and likely modification before acceptance.",
                RiskLevel.ORANGE: "This represents a MEDIUM-HIGH RISK that should be carefully reviewed and potentially modified.",
                RiskLevel.YELLOW: "This represents a LOW-MEDIUM RISK that warrants attention but may be acceptable with proper understanding.",
                RiskLevel.WHITE: "This represents a LOW RISK with standard terms that are generally acceptable."
            }
            
            rationale_parts.append(risk_explanations.get(risk_level, "Risk level assessment completed."))
            
            # Evidence-based insights
            if evidence_count > 0:
                rationale_parts.append(
                    f"This assessment is supported by analysis of {evidence_count} similar legal precedents and documents."
                )
                
                # Add specific evidence insights
                high_similarity_count = sum(1 for e in evidence if e.similarity > 0.85)
                if high_similarity_count > 0:
                    rationale_parts.append(
                        f"Found {high_similarity_count} highly similar cases that strongly inform this assessment."
                    )
            
            # Clause-specific advice
            clause_advice = {
                ClauseTag.LIABILITY: "Consider liability caps, exclusions, and mutual liability provisions.",
                ClauseTag.INDEMNITY: "Review scope of indemnification and ensure reciprocal terms where appropriate.",
                ClauseTag.TERMINATION: "Verify termination notice periods and post-termination obligations.",
                ClauseTag.PAYMENT: "Check payment terms, late fees, and dispute resolution procedures.",
                ClauseTag.IP: "Ensure proper intellectual property ownership and licensing terms.",
                ClauseTag.CONFIDENTIALITY: "Verify confidentiality scope and disclosure exceptions.",
                ClauseTag.GOVERNING_LAW: "Confirm acceptable jurisdiction and governing law.",
                ClauseTag.ARBITRATION: "Review arbitration procedures and venue requirements."
            }
            
            if clause.tag in clause_advice:
                rationale_parts.append(clause_advice[clause.tag])
            
            return " ".join(rationale_parts)
            
        except Exception as e:
            logger.warning(f"Rationale generation failed: {e}")
            return self._create_fallback_rationale(clause)
    
    def _create_fallback_rationale(self, clause: Clause) -> str:
        """Create basic fallback rationale when detailed analysis fails"""
        clause_type = clause.tag.value.replace("_", " ")
        return (
            f"This {clause_type} clause requires careful review. "
            f"Legal precedents and standard practices should be consulted. "
            f"Consider seeking professional legal advice for complex terms."
        )

class RiskAggregator:
    """Aggregates and analyzes risks across multiple clauses"""
    
    def __init__(self):
        self.risk_calculator = RiskCalculator()
    
    async def assess_document_risks(
        self, 
        clauses: List[Clause]
    ) -> Tuple[List[RiskScore], Dict[str, Any]]:
        """
        Assess risks for all clauses in a document
        
        Returns:
            Tuple of (individual_risk_scores, document_risk_summary)
        """
        try:
            logger.info(f"Assessing risks for {len(clauses)} clauses")
            
            # Process all clauses concurrently (with limit to avoid overwhelming)
            semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent assessments
            
            async def assess_single_clause(clause: Clause) -> RiskScore:
                async with semaphore:
                    return await self._assess_clause_risk(clause)
            
            # Execute risk assessments
            risk_scores = await asyncio.gather(
                *[assess_single_clause(clause) for clause in clauses],
                return_exceptions=True
            )
            
            # Filter out exceptions and log them
            valid_risk_scores = []
            for i, result in enumerate(risk_scores):
                if isinstance(result, Exception):
                    logger.error(f"Risk assessment failed for clause {clauses[i].id}: {result}")
                    # Create fallback risk score
                    fallback_risk = await self._create_fallback_risk_score(clauses[i])
                    valid_risk_scores.append(fallback_risk)
                else:
                    valid_risk_scores.append(result)
            
            # Generate document-level risk summary
            risk_summary = self._generate_risk_summary(valid_risk_scores)
            
            logger.info(
                f"Risk assessment completed: {len(valid_risk_scores)} scores generated, "
                f"overall risk level: {risk_summary.get('overall_risk_level')}"
            )
            
            return valid_risk_scores, risk_summary
            
        except Exception as e:
            logger.error(f"Document risk assessment failed: {e}")
            # Return minimal fallback
            fallback_scores = [
                await self._create_fallback_risk_score(clause) 
                for clause in clauses
            ]
            fallback_summary = {"overall_risk_level": "unknown", "error": str(e)}
            return fallback_scores, fallback_summary
    
    async def _assess_clause_risk(self, clause: Clause) -> RiskScore:
        """Assess risk for a single clause and fetch mitigation advice"""
        try:
            # 1  core scoring --------------------------------------------------
            risk_score, risk_level, rationale = (
                await self.risk_calculator.calculate_clause_risk(clause)
            )
            evidence = await self.risk_calculator._retrieve_clause_evidence(clause)


            # 3  build result --------------------------------------------------
            return RiskScore(
                clause_id          = clause.id,
                level              = risk_level,
                score              = risk_score,
                rationale          = rationale,
                supporting_context = evidence,  # ← new field
            )

        except Exception as err:
            logger.error("Clause %s risk assessment failed: %s", clause.id, err)
            return await self._create_fallback_risk_score(clause)
    
    async def _create_fallback_risk_score(self, clause: Clause) -> RiskScore:
        """Create fallback risk score when assessment fails"""
        base_risk = self.risk_calculator.config.base_risk_weights.get(clause.tag, 0.3)
        risk_level = self.risk_calculator._determine_risk_level(base_risk)
        
        return RiskScore(
            clause_id=clause.id,
            level=risk_level,
            score=base_risk,
            rationale=f"Basic risk assessment for {clause.tag.value} clause. Detailed analysis unavailable.",
            supporting_context=[]
        )
    
    def _generate_risk_summary(self, risk_scores: List[RiskScore]) -> Dict[str, Any]:
        """Generate document-level risk summary"""
        if not risk_scores:
            return {"overall_risk_level": "unknown", "total_clauses": 0}
        
        # Count risks by level
        risk_counts = defaultdict(int)
        total_score = 0.0
        
        for risk in risk_scores:
            risk_counts[risk.level.value] += 1
            total_score += risk.score
        
        # Calculate overall risk level
        red_count = risk_counts.get("red", 0)
        orange_count = risk_counts.get("orange", 0)
        yellow_count = risk_counts.get("yellow", 0)
        
        if red_count > 0:
            overall_level = "red"
        elif orange_count >= 2 or (orange_count >= 1 and yellow_count >= 2):
            overall_level = "orange"
        elif orange_count >= 1 or yellow_count >= 3:
            overall_level = "yellow"
        else:
            overall_level = "white"
        
        average_score = total_score / len(risk_scores)
        
        return {
            "overall_risk_level": overall_level,
            "average_risk_score": round(average_score, 2),
            "total_clauses": len(risk_scores),
            "risk_distribution": dict(risk_counts),
            "high_risk_clauses": red_count + orange_count,
            "requires_review": red_count > 0 or orange_count >= 2
        }

# Global service instance
_risk_service = RiskAggregator()

# Public API functions for router integration
async def score_risks(clauses: List[Clause]) -> List[RiskScore]:
    """
    Score risks for a list of clauses with RAG evidence
    
    Args:
        clauses: List of identified legal clauses
        
    Returns:
        List of risk scores with evidence and rationales
    """
    try:
        risk_scores, _ = await _risk_service.assess_document_risks(clauses)
        return risk_scores
    except Exception as e:
        logger.error(f"Risk scoring failed: {e}")
        # Return basic risk scores as fallback
        return [
            await _risk_service._create_fallback_risk_score(clause)
            for clause in clauses
        ]

async def assess_document_risks(clauses: List[Clause]) -> Tuple[List[RiskScore], Dict[str, Any]]:
    """
    Comprehensive document risk assessment
    
    Returns:
        Tuple of (risk_scores, document_summary)
    """
    return await _risk_service.assess_document_risks(clauses)

async def calculate_clause_risk(clause: Clause) -> RiskScore:
    """
    Calculate risk for a single clause
    
    Args:
        clause: Individual legal clause
        
    Returns:
        Risk score with evidence and rationale
    """
    return await _risk_service._assess_clause_risk(clause)

async def health_check() -> Dict[str, Any]:
    """Health check for risk service"""
    try:
        # Test basic functionality
        test_clause = Clause(
            id="test",
            tag=ClauseTag.OTHER,
            text="Test clause for health check",
            span=None
        )
        
        # Quick risk calculation test
        risk_score = await calculate_clause_risk(test_clause)
        
        return {
            "status": "healthy",
            "service": "risk_scoring",
            "test_result": "passed" if risk_score else "failed",
            "features": [
                "rag_enhanced_scoring",
                "evidence_analysis",
                "multi_level_assessment", 
                "document_summarization"
            ]
        }
    except Exception as e:
        return {
            "status": "degraded",
            "service": "risk_scoring", 
            "error": str(e),
            "fallback_available": True
        }
