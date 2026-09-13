"""
Tests for evidence models and comparison.
"""
import pytest
from evidence.models import (
    Source, Claim, Evidence, EvidenceComparison, Conflict, ConflictType, NLILabel
)
from evidence.comparator import EvidenceComparator
from evidence.disagreement_detector import MockNLIDetector, get_nli_detector


class TestEvidenceModels:
    """Test evidence data models."""
    
    def test_source_creation(self):
        """Test Source model creation."""
        source = Source(
            url="https://example.com/article",
            title="Test Article",
            snippet="This is a test snippet",
        )
        assert source.url == "https://example.com/article"
        assert source.title == "Test Article"
        assert source.snippet == "This is a test snippet"
    
    def test_claim_creation(self):
        """Test Claim model creation."""
        source = Source(url="https://example.com", title="Test", snippet="Snippet")
        claim = Claim(text="This is a test claim", source=source)
        assert claim.text == "This is a test claim"
        assert claim.source.url == "https://example.com"
        assert claim.id is not None
    
    def test_evidence_creation(self):
        """Test Evidence model creation."""
        source = Source(url="https://example.com", title="Test", snippet="Snippet")
        claim = Claim(text="Test claim", source=source)
        
        evidence = Evidence(
            agent_id="agent_1",
            query="What is testing?",
            answer="Testing is important",
            claims=[claim],
            sources=[source],
        )
        
        assert evidence.agent_id == "agent_1"
        assert evidence.query == "What is testing?"
        assert len(evidence.claims) == 1
        assert len(evidence.sources) == 1
    
    def test_evidence_comparison(self):
        """Test EvidenceComparison model."""
        comparison = EvidenceComparison(
            agent_a_id="agent_a",
            agent_b_id="agent_b",
            has_disagreement=True,
            contradiction_score=0.85,
        )
        
        assert comparison.has_disagreement is True
        assert comparison.contradiction_score == 0.85
    
    def test_conflict_types(self):
        """Test ConflictType enum values."""
        assert ConflictType.FACTUAL.value == "factual"
        assert ConflictType.TEMPORAL.value == "temporal"
        assert ConflictType.VERSION.value == "version"
        assert ConflictType.CONTEXTUAL.value == "contextual"
        assert ConflictType.SOURCE.value == "source"
    
    def test_nli_labels(self):
        """Test NLILabel enum values."""
        assert NLILabel.ENTAILMENT.value == "entailment"
        assert NLILabel.CONTRADICTION.value == "contradiction"
        assert NLILabel.NEUTRAL.value == "neutral"


class TestMockNLIDetector:
    """Test the Mock NLI detector."""
    
    def test_contradiction_detection(self):
        """Test detection of contradictions."""
        detector = MockNLIDetector()
        
        # Test temporal contradiction
        result = detector.predict(
            "The policy was introduced in 2019",
            "The policy was introduced in 2020"
        )
        assert result["label"] == NLILabel.CONTRADICTION
        assert result["score"] > 0.8
    
    def test_entailment_detection(self):
        """Test detection of entailment."""
        detector = MockNLIDetector()
        
        result = detector.predict(
            "Python is a programming language used for web development",
            "Python is used for building web applications"
        )
        assert result["label"] == NLILabel.ENTAILMENT
    
    def test_neutral_detection(self):
        """Test detection of neutral relationships."""
        detector = MockNLIDetector()
        
        result = detector.predict(
            "The weather is sunny today",
            "Python was created in 1991"
        )
        assert result["label"] == NLILabel.NEUTRAL
    
    def test_disagreement_detection(self):
        """Test the disagreement detection method."""
        detector = MockNLIDetector(threshold=0.7)
        
        result = detector.detect_disagreement(
            "The price increased to $100",
            "The price decreased to $50"
        )
        
        assert "disagreement" in result
        assert "contradiction_score" in result
        assert "threshold" in result


class TestEvidenceComparator:
    """Test the EvidenceComparator."""
    
    def test_comparator_agreement(self):
        """Test comparator when evidence agrees."""
        detector = MockNLIDetector()
        comparator = EvidenceComparator(detector)
        
        source = Source(url="https://example.com", title="Test", snippet="Snippet")
        
        evidence_a = Evidence(
            agent_id="agent_a",
            query="Test query",
            answer="Answer A",
            claims=[Claim(text="Python is a programming language", source=source)],
            sources=[source],
        )
        
        evidence_b = Evidence(
            agent_id="agent_b",
            query="Test query",
            answer="Answer B",
            claims=[Claim(text="Python is used for programming", source=source)],
            sources=[source],
        )
        
        comparison = comparator.compare(evidence_a, evidence_b)
        
        assert comparison.has_disagreement is False
        assert comparison.contradiction_score < 0.7
    
    def test_comparator_disagreement(self):
        """Test comparator when evidence disagrees."""
        detector = MockNLIDetector()
        comparator = EvidenceComparator(detector)
        
        source_a = Source(url="https://example.com/a", title="Source A", snippet="Snippet")
        source_b = Source(url="https://example.com/b", title="Source B", snippet="Snippet")
        
        evidence_a = Evidence(
            agent_id="agent_a",
            query="Test query",
            answer="Answer A",
            claims=[Claim(text="The policy was introduced in 2019", source=source_a)],
            sources=[source_a],
        )
        
        evidence_b = Evidence(
            agent_id="agent_b",
            query="Test query",
            answer="Answer B",
            claims=[Claim(text="The policy was introduced in 2020", source=source_b)],
            sources=[source_b],
        )
        
        comparison = comparator.compare(evidence_a, evidence_b)
        
        assert comparison.has_disagreement is True
        assert comparison.contradiction_score >= 0.7


class TestRealNLIDetector:
    """Test real NLI detector (requires model loading)."""
    
    @pytest.mark.slow
    def test_detector_loading(self):
        """Test that the NLI detector can be loaded."""
        detector = get_nli_detector()
        assert detector is not None
    
    @pytest.mark.slow
    def test_real_prediction(self):
        """Test real NLI prediction."""
        detector = get_nli_detector()
        
        result = detector.predict(
            "The sky is blue",
            "The sky is green"
        )
        
        assert "label" in result
        assert "score" in result
        assert result["label"] in [NLILabel.CONTRADICTION, NLILabel.NEUTRAL, NLILabel.ENTAILMENT]