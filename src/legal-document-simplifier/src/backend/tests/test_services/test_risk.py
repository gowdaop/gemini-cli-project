import pytest
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from src.backend.services import risk
from src.backend.schemas.analysis import Clause, ClauseTag, RiskLevel, PageSpan, RAGContextItem

@pytest.fixture
def sample_rag_contexts():
    """Sample RAG context items for testing"""
    return [
        RAGContextItem(
            chunk_id=123,
            content="Sample legal precedent about indemnification clauses and risk assessment.",
            doc_type="legal_document",
            jurisdiction="US",
            date="2023-01-01",
            source_url="test://sample-doc-1.pdf",
            similarity=0.95
        ),
        RAGContextItem(
            chunk_id=456,
            content="Another legal precedent discussing liability limitations in contracts.",
            doc_type="legal_precedent",
            jurisdiction="UK",
            date="2022-06-15",
            source_url="test://sample-doc-2.pdf",
            similarity=0.87
        )
    ]

@pytest.fixture
def sample_clause():
    """Sample clause for testing"""
    return Clause(
        id="c-0001",
        tag=ClauseTag.INDEMNITY,
        text="The Company shall indemnify and hold harmless the Client from any claims or damages.",
        span=PageSpan(page=1, start_line=1, end_line=2)
    )

class TestRiskCalculator:
    
    @pytest.fixture
    def risk_calculator(self):
        return risk.RiskCalculator()
    
    @pytest.mark.asyncio
    async def test_calculate_clause_risk_liability(self, risk_calculator, sample_clause):
        """Test risk calculation for liability clause"""
        liability_clause = Clause(
            id="c-001",
            tag=ClauseTag.LIABILITY,
            text="Company shall not be liable for any indirect damages",
            span=PageSpan(page=1, start_line=1, end_line=1)
        )
        
        with patch.object(risk_calculator, '_retrieve_clause_evidence') as mock_evidence:
            mock_evidence.return_value = []
            
            score, level, rationale = await risk_calculator.calculate_clause_risk(liability_clause)
            
            assert 0.0 <= score <= 1.0
            assert isinstance(level, RiskLevel)
            assert len(rationale) > 0
            assert "liability" in rationale.lower()

    @pytest.mark.asyncio
    async def test_calculate_clause_risk_with_evidence(self, risk_calculator, sample_clause, sample_rag_contexts):
        """Test risk calculation with RAG evidence"""
        with patch.object(risk_calculator, '_retrieve_clause_evidence') as mock_evidence:
            mock_evidence.return_value = sample_rag_contexts
            
            score, level, rationale = await risk_calculator.calculate_clause_risk(sample_clause)
            
            assert score > 0.0
            assert "similar" in rationale.lower() or "precedent" in rationale.lower()

    @pytest.mark.asyncio 
    async def test_retrieve_clause_evidence(self, risk_calculator, sample_clause, sample_rag_contexts):
        """Test evidence retrieval for clause"""
        with patch('src.backend.services.risk.rag.retrieve_contexts') as mock_retrieve:
            mock_retrieve.return_value = sample_rag_contexts
            
            evidence = await risk_calculator._retrieve_clause_evidence(sample_clause)
            
            assert len(evidence) >= 0
            mock_retrieve.assert_called_once()

    def test_create_risk_query(self, risk_calculator, sample_clause):
        """Test risk query creation"""
        query = risk_calculator._create_risk_query(sample_clause)
        
        assert len(query) > 0
        assert "indemnify" in query.lower()
        assert "risk" in query.lower()

    def test_analyze_risk_indicators(self, risk_calculator, sample_rag_contexts):
        """Test risk indicator analysis"""
        # Add high-risk content to test
        sample_rag_contexts[0].content = "unlimited liability without limitation severe penalty"
        
        risk_score = risk_calculator._analyze_risk_indicators(sample_rag_contexts)
        
        assert risk_score >= 0.0

# Rest of your test classes...
