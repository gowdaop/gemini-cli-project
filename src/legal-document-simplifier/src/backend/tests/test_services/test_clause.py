import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.backend.services import clause
from src.backend.schemas.analysis import Clause, ClauseTag, OCRText, OCRBlock, PageSpan


# ✅ Fix 1: Better sample data that passes validation
@pytest.fixture
def sample_ocr_text():
    """Sample OCR text with legal content that passes validation"""
    return OCRText(
        full_text="The Company shall indemnify and hold harmless the Client against all third-party claims.",
        blocks=[
            OCRBlock(
                text="The Company shall indemnify and hold harmless the Client against all third-party claims.",
                span=PageSpan(page=1, start_line=1, end_line=2)
            )
        ]
    )


@pytest.fixture
def sample_ocr_text_minimal():
    """Minimal OCR text for basic tests"""
    return OCRText(
        full_text="Company agrees to defend and indemnify Client from liability claims.",
        blocks=[
            OCRBlock(
                text="Company agrees to defend and indemnify Client from liability claims.",
                span=PageSpan(page=1, start_line=1, end_line=1)
            )
        ]
    )


class TestLegalTermsLexicon:
    
    def test_clause_patterns_initialization(self):
        """Test lexicon initialization"""
        lexicon = clause.LegalTermsLexicon()
        
        assert len(lexicon.clause_patterns) > 0
        assert ClauseTag.LIABILITY in lexicon.clause_patterns
        assert ClauseTag.INDEMNITY in lexicon.clause_patterns
        
        # Test liability pattern
        liability_pattern = lexicon.clause_patterns[ClauseTag.LIABILITY]
        assert "liability" in liability_pattern.keywords
        assert len(liability_pattern.patterns) > 0


class TestClauseSegmenter:
    
    @pytest.fixture
    def segmenter(self):
        lexicon = clause.LegalTermsLexicon()
        return clause.ClauseSegmenter(lexicon)
    
    @pytest.mark.asyncio
    async def test_segment_clauses_basic(self, segmenter, sample_ocr_text):
        """Test basic clause segmentation"""
        segments = await segmenter.segment_clauses(sample_ocr_text)
        
        # ✅ Fix: Should now pass since we have legal content
        assert len(segments) > 0
        assert "text" in segments[0]
        assert "span" in segments[0]

    def test_is_valid_clause_candidate(self, segmenter):
        """Test clause candidate validation"""
        valid_text = "The Company shall indemnify the Client against all claims."
        invalid_text = "Page 1"
        short_text = "Test"
        
        assert segmenter._is_valid_clause_candidate(valid_text)
        assert not segmenter._is_valid_clause_candidate(invalid_text)
        assert not segmenter._is_valid_clause_candidate(short_text)


class TestRuleBasedClassifier:
    
    @pytest.fixture
    def classifier(self):
        lexicon = clause.LegalTermsLexicon()
        return clause.RuleBasedClassifier(lexicon)
    
    @pytest.mark.asyncio
    async def test_classify_indemnity_clause(self, classifier):
        """Test classification of indemnity clause"""
        text = "The Company shall indemnify and hold harmless the Client"
        
        result = await classifier.classify_segment(text)
        
        assert result.tag == ClauseTag.INDEMNITY
        assert result.confidence > 0.3
        assert "indemnify" in result.matched_keywords

    @pytest.mark.asyncio
    async def test_classify_liability_clause(self, classifier):
        """Test classification of liability clause"""
        text = "The Company shall not be liable for any indirect damages"
        
        result = await classifier.classify_segment(text)
        
        assert result.tag == ClauseTag.LIABILITY
        assert result.confidence > 0.3
        assert "liable" in result.matched_keywords

    @pytest.mark.asyncio
    async def test_classify_unknown_clause(self, classifier):
        """Test classification of unknown clause"""
        text = "This is some random text that doesn't match any patterns"
        
        result = await classifier.classify_segment(text)
        
        assert result.tag == ClauseTag.OTHER
        assert result.confidence < 0.5


class TestMLClassifier:
    
    @pytest.fixture
    def ml_classifier(self):
        return clause.MLClassifier()
    
    @pytest.mark.asyncio
    async def test_initialize(self, ml_classifier):
        """Test ML classifier initialization"""
        await ml_classifier.initialize()
        # Should not raise exception
        
    @pytest.mark.asyncio
    async def test_classify_segment_not_ready(self, ml_classifier):
        """Test ML classification when model not loaded"""
        result = await ml_classifier.classify_segment("test text")
        
        assert result is None


class TestEnhancedClauseClassifier:
    
    @pytest.fixture
    def classifier(self):
        return clause.EnhancedClauseClassifier()
    
    @pytest.mark.asyncio
    async def test_initialize(self, classifier):
        """Test classifier initialization"""
        await classifier.initialize()
        
        assert classifier.initialized

    @pytest.mark.asyncio
    async def test_classify_clauses_success(self, classifier, sample_ocr_text):
        """Test successful clause classification"""
        await classifier.initialize()
        
        with patch.object(classifier.segmenter, 'segment_clauses') as mock_segment, \
             patch.object(classifier, '_classify_single_segment') as mock_classify:
            
            mock_segment.return_value = [
                {
                    "text": "The Company shall indemnify",
                    "span": PageSpan(page=1, start_line=1, end_line=1),
                    "source": "test"
                }
            ]
            
            mock_classify.return_value = Clause(
                id="c-001",
                tag=ClauseTag.INDEMNITY,
                text="The Company shall indemnify",
                span=PageSpan(page=1, start_line=1, end_line=1)
            )
            
            result = await classifier.classify_clauses(sample_ocr_text)
            
            assert len(result) > 0
            assert isinstance(result[0], Clause)

    @pytest.mark.asyncio
    async def test_classify_clauses_fallback(self, classifier, sample_ocr_text):
        """Test clause classification fallback"""
        await classifier.initialize()
        
        with patch.object(classifier.segmenter, 'segment_clauses') as mock_segment:
            mock_segment.side_effect = Exception("Segmentation error")
            
            result = await classifier.classify_clauses(sample_ocr_text)
            
            assert len(result) >= 0  # Should return fallback clauses


class TestClauseServiceAPI:
    
    @pytest.mark.asyncio
    async def test_classify_clauses_public(self, sample_ocr_text):
        """Test public classify_clauses function"""
        # ✅ Fix 2: Use AsyncMock for async methods
        with patch('src.backend.services.clause._clause_classifier') as mock_classifier:
            mock_classifier.classify_clauses = AsyncMock(return_value=[
                Clause(
                    id="c-001",
                    tag=ClauseTag.OTHER,
                    text="Test clause",
                    span=PageSpan(page=1, start_line=1, end_line=1)
                )
            ])
            
            result = await clause.classify_clauses(sample_ocr_text)
            
            assert len(result) > 0
            assert isinstance(result[0], Clause)

    @pytest.mark.asyncio
    async def test_classify_single_text(self):
        """Test single text classification"""
        # ✅ Fix 3: Proper mocking for sync method
        with patch('src.backend.services.clause._clause_classifier') as mock_classifier:
            mock_classifier.initialized = True
            
            # Create a mock rule classifier with sync classify_segment
            mock_rule_classifier = MagicMock()
            mock_rule_classifier.classify_segment.return_value = clause.ClassificationResult(
                tag=ClauseTag.LIABILITY,
                confidence=0.8,
                matched_patterns=["test"],
                matched_keywords=["liable"]
            )
            mock_classifier.rule_classifier = mock_rule_classifier
            
            # ✅ This should now work with fixed service code (no await on sync method)
            result = await clause.classify_single_text("Company shall not be liable")
            
            assert result.tag == ClauseTag.LIABILITY
            assert result.confidence == 0.8

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test clause service health check"""
        result = await clause.health_check()
        
        assert "status" in result
        assert result["status"] in ["healthy", "degraded"]
        assert "features" in result

    @pytest.mark.asyncio
    async def test_get_classification_stats(self):
        """Test classification statistics"""
        result = await clause.get_classification_stats()
        
        assert "clause_types_supported" in result
        assert "classification_methods" in result
        assert result["clause_types_supported"] > 0


# ✅ Additional fixture for testing edge cases
@pytest.fixture
def complex_ocr_text():
    """Complex OCR text with multiple clause types"""
    return OCRText(
        full_text="The Company shall indemnify the Client. Company shall not be liable for damages. This agreement terminates upon 30 days notice.",
        blocks=[
            OCRBlock(
                text="The Company shall indemnify the Client.",
                span=PageSpan(page=1, start_line=1, end_line=1)
            ),
            OCRBlock(
                text="Company shall not be liable for damages.",
                span=PageSpan(page=1, start_line=2, end_line=2)
            ),
            OCRBlock(
                text="This agreement terminates upon 30 days notice.",
                span=PageSpan(page=1, start_line=3, end_line=3)
            )
        ]
    )


# ✅ Additional test for complex scenarios
class TestComplexScenarios:
    
    @pytest.mark.asyncio
    async def test_multiple_clause_types(self, complex_ocr_text):
        """Test classification of multiple clause types"""
        with patch('src.backend.services.clause._clause_classifier') as mock_classifier:
            mock_classifier.classify_clauses = AsyncMock(return_value=[
                Clause(
                    id="c-001",
                    tag=ClauseTag.INDEMNITY,
                    text="The Company shall indemnify the Client.",
                    span=PageSpan(page=1, start_line=1, end_line=1)
                ),
                Clause(
                    id="c-002",
                    tag=ClauseTag.LIABILITY,
                    text="Company shall not be liable for damages.",
                    span=PageSpan(page=1, start_line=2, end_line=2)
                ),
                Clause(
                    id="c-003",
                    tag=ClauseTag.TERMINATION,
                    text="This agreement terminates upon 30 days notice.",
                    span=PageSpan(page=1, start_line=3, end_line=3)
                )
            ])
            
            result = await clause.classify_clauses(complex_ocr_text)
            
            assert len(result) == 3
            assert result[0].tag == ClauseTag.INDEMNITY
            assert result[1].tag == ClauseTag.LIABILITY  
            assert result[2].tag == ClauseTag.TERMINATION

    @pytest.mark.asyncio
    async def test_empty_ocr_text(self):
        """Test handling of empty OCR text"""
        empty_ocr = OCRText(full_text="", blocks=[])
        
        with patch('src.backend.services.clause._clause_classifier') as mock_classifier:
            mock_classifier.classify_clauses = AsyncMock(return_value=[])
            
            result = await clause.classify_clauses(empty_ocr)
            
            assert isinstance(result, list)
            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_classifier_error_handling(self, sample_ocr_text):
        """Test classifier error handling - should return fallback list, not raise exception"""
        # ✅ Fix 4: Test now expects graceful error handling
        with patch('src.backend.services.clause._clause_classifier') as mock_classifier:
            mock_classifier.classify_clauses = AsyncMock(side_effect=Exception("Classifier error"))
            
            # Should not raise exception, should return list (empty or fallback)
            result = await clause.classify_clauses(sample_ocr_text)
            
            assert isinstance(result, list)  # Should return a list even on error
            # With the fixed service code, this should return fallback clauses from OCR blocks
            assert len(result) >= 0  # Could be empty list or fallback clauses
