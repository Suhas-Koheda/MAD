"""
Tests for conflict classification.
"""
import pytest
from conflict.classifier import ConflictClassifier, get_conflict_classifier
from evidence.models import ConflictType


class TestConflictClassifier:
    """Test conflict type classification."""
    
    def test_temporal_classification(self):
        """Test temporal conflict classification."""
        classifier = ConflictClassifier()
        
        result = classifier.classify(
            claim_a="The policy was introduced in 2019",
            claim_b="The policy was introduced in 2020",
            query="When was the policy introduced?"
        )
        
        assert result["type"] == ConflictType.TEMPORAL.value
        assert result["confidence"] > 0.8
    
    def test_version_classification(self):
        """Test version conflict classification."""
        classifier = ConflictClassifier()
        
        result = classifier.classify(
            claim_a="Version 1.0 supports Python 3.8",
            claim_b="Version 2.0 supports Python 3.10",
            query="Which Python versions are supported?"
        )
        
        assert result["type"] == ConflictType.VERSION.value
        assert result["confidence"] > 0.8
    
    def test_contextual_classification(self):
        """Test contextual conflict classification."""
        classifier = ConflictClassifier()
        
        result = classifier.classify(
            claim_a="The drug is effective for adults",
            claim_b="The drug is not effective for children",
            query="Is the drug effective?"
        )
        
        assert result["type"] == ConflictType.CONTEXTUAL.value
        assert result["confidence"] > 0.7
    
    def test_factual_classification(self):
        """Test factual conflict classification."""
        classifier = ConflictClassifier()
        
        result = classifier.classify(
            claim_a="The capital of France is Paris",
            claim_b="The capital of France is Lyon",
            query="What is the capital of France?"
        )
        
        assert result["type"] == ConflictType.FACTUAL.value
        assert result["confidence"] > 0.7
    
    def test_classification_result_structure(self):
        """Test that classification results have correct structure."""
        classifier = ConflictClassifier()
        
        result = classifier.classify(
            claim_a="Test claim A",
            claim_b="Test claim B",
            query="Test query"
        )
        
        assert "type" in result
        assert "confidence" in result
        assert "explanation" in result
        assert isinstance(result["type"], str)
        assert 0 <= result["confidence"] <= 1


class TestConflictClassifierWithLLM:
    """Test conflict classifier with LLM client."""
    
    def test_llm_classification(self):
        """Test classification using LLM."""
        from llm_client import DemoLLMClient
        
        llm_client = DemoLLMClient()
        classifier = ConflictClassifier(llm_client)
        
        result = classifier.classify(
            claim_a="The event happened in 2022",
            claim_b="The event happened in 2024",
            query="When did the event happen?"
        )
        
        assert "type" in result
        assert "confidence" in result


class TestConflictClassifierEdgeCases:
    """Test edge cases for conflict classification."""
    
    def test_empty_claims(self):
        """Test classification with empty claims."""
        classifier = ConflictClassifier()
        
        result = classifier.classify(
            claim_a="",
            claim_b="",
            query="Test query"
        )
        
        assert result["type"] in [ct.value for ct in ConflictType]
    
    def test_long_claims(self):
        """Test classification with long claims."""
        classifier = ConflictClassifier()
        
        long_claim_a = "This is a very long claim. " * 100
        long_claim_b = "This is another very long claim. " * 100
        
        result = classifier.classify(
            claim_a=long_claim_a,
            claim_b=long_claim_b,
            query="Test query"
        )
        
        assert "type" in result
    
    def test_different_languages(self):
        """Test classification with non-English text."""
        classifier = ConflictClassifier()
        
        result = classifier.classify(
            claim_a="2019年に導入された政策",
            claim_b="2020年に導入された政策",
            query="政策はいつ導入されたか？"
        )
        
        # Should still return a valid type
        assert result["type"] in [ct.value for ct in ConflictType]