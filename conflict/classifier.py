"""
Conflict type classifier for classifying evidence disagreements.
Uses LLM-based classification with structured output.
"""
import json
import re
from typing import Optional, Dict, Any
from loguru import logger

from evidence.models import ConflictType, Conflict, Claim
from config import settings


CONFLICT_TYPE_PROMPT = """You are a conflict type classifier. Given two conflicting claims, classify the type of conflict.

## Conflict Types

1. FACTUAL - Two sources make mutually incompatible factual claims about the same thing.
   Example: "Water boils at 100°C" vs "Water boils at 90°C"

2. TEMPORAL - The claims differ because they refer to different points in time.
   Example: "The policy was introduced in 2019" vs "The policy was introduced in 2020"

3. VERSION - The claims refer to different software/product/document/policy versions.
   Example: "Version 1.0 supports X" vs "Version 2.0 supports Y"

4. CONTEXTUAL - The claims differ because their conditions or contexts differ.
   Example: "The drug is effective for adults" vs "The drug is not effective for children"

5. SOURCE - The disagreement is primarily caused by differences in source authority/reliability/provenance.
   Example: A blog says X vs a peer-reviewed paper says Y

## Input

Claim A: {claim_a}
Claim B: {claim_b}

Query: {query}

## Response Format

Return ONLY a JSON object with this exact structure:
{{
    "type": "FACTUAL|TEMPORAL|VERSION|CONTEXTUAL|SOURCE",
    "confidence": 0.0-1.0,
    "explanation": "Brief explanation of why this conflict type was chosen"
}}
"""


class ConflictClassifier:
    """Classifies conflicts between evidence claims using LLM."""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.demo_mode = settings.demo_mode
    
    def classify(
        self,
        claim_a: str,
        claim_b: str,
        query: str = "",
    ) -> Dict[str, Any]:
        """
        Classify the type of conflict between two claims.
        
        Args:
            claim_a: First conflicting claim
            claim_b: Second conflicting claim
            query: Original user query
            
        Returns:
            Dictionary with type, confidence, and explanation
        """
        if self.demo_mode or not self.llm_client:
            return self._classify_by_patterns(claim_a, claim_b, query)
        
        return self._classify_by_llm(claim_a, claim_b, query)
    
    def _classify_by_llm(
        self,
        claim_a: str,
        claim_b: str,
        query: str,
    ) -> Dict[str, Any]:
        """Use LLM to classify conflict type."""
        try:
            prompt = CONFLICT_TYPE_PROMPT.format(
                claim_a=claim_a,
                claim_b=claim_b,
                query=query,
            )
            
            response = self.llm_client.chat(prompt)
            
            # Parse JSON response
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                # Validate and normalize
                conflict_type = result.get("type", "UNKNOWN").upper()
                valid_types = [t.value for t in ConflictType]
                if conflict_type.lower() not in valid_types:
                    conflict_type = ConflictType.UNKNOWN.value
                else:
                    conflict_type = conflict_type.lower()
                
                return {
                    "type": conflict_type,
                    "confidence": float(result.get("confidence", 0.8)),
                    "explanation": result.get("explanation", "LLM classification"),
                }
            
            # Fallback if JSON parsing fails
            return self._classify_by_patterns(claim_a, claim_b, query)
            
        except Exception as e:
            logger.error(f"LLM conflict classification failed: {e}")
            return self._classify_by_patterns(claim_a, claim_b, query)
    
    def _classify_by_patterns(
        self,
        claim_a: str,
        claim_b: str,
        query: str,
    ) -> Dict[str, Any]:
        """
        Pattern-based conflict classification for demo mode.
        Uses heuristics and keyword matching.
        """
        claim_a_lower = claim_a.lower()
        claim_b_lower = claim_b.lower()
        
        # Check for TEMPORAL conflict (dates, years, time references)
        year_pattern = r'\b(?:19|20)\d{2}\b'
        years_a = re.findall(year_pattern, claim_a)
        years_b = re.findall(year_pattern, claim_b)
        
        if years_a and years_b:
            if set(years_a) != set(years_b):
                return {
                    "type": ConflictType.TEMPORAL.value,
                    "confidence": 0.88,
                    "explanation": f"Claims refer to different years: {', '.join(years_a)} vs {', '.join(years_b)}",
                }
        
        # Check for VERSION conflict
        version_patterns = [
            r'v(?:ersion)?\s*\d+\.\d+',
            r'v(?:ersion)?\s*\d+(?:\.\d+)?',
            r'beta|alpha|release|draft|final',
            r'legacy|old|new|latest|current',
        ]
        
        for pattern in version_patterns:
            if re.search(pattern, claim_a_lower) and re.search(pattern, claim_b_lower):
                if claim_a_lower != claim_b_lower:
                    return {
                        "type": ConflictType.VERSION.value,
                        "confidence": 0.85,
                        "explanation": "Claims refer to different versions or releases",
                    }
        
        # Check for CONTEXTUAL conflict (conditions, qualifications)
        context_words = [
            ["but", "however", "except", "although", "unless"],
            ["enterprise", "individual", "consumer", "business"],
            ["adult", "child", "elderly", "young"],
            ["rarely", "often", "always", "never"],
            ["may", "will", "must", "should"],
        ]
        
        for context_group in context_words:
            for word in context_group:
                if word in claim_a_lower or word in claim_b_lower:
                    return {
                        "type": ConflictType.CONTEXTUAL.value,
                        "confidence": 0.82,
                        "explanation": f"Claims contain different qualifying conditions",
                    }
        
        # Check for SOURCE conflict
        source_words = [
            ["unofficial", "official"],
            ["blog", "peer-reviewed"],
            ["anecdotal", "statistical"],
            ["reported", "confirmed"],
        ]
        
        for source_group in source_words:
            for word in source_group:
                if word in claim_a_lower or word in claim_b_lower:
                    return {
                        "type": ConflictType.SOURCE.value,
                        "confidence": 0.80,
                        "explanation": "Claims differ in source credibility or provenance",
                    }
        
        # Default to FACTUAL conflict
        return {
            "type": ConflictType.FACTUAL.value,
            "confidence": 0.75,
            "explanation": "Direct factual contradiction detected",
        }
    
    def classify_from_conflict(
        self,
        conflict: Conflict,
        query: str = "",
    ) -> Dict[str, Any]:
        """Classify from a Conflict model instance."""
        return self.classify(
            claim_a=conflict.claim_a.text,
            claim_b=conflict.claim_b.text,
            query=query,
        )


def get_conflict_classifier(llm_client=None) -> ConflictClassifier:
    """Get or create a conflict classifier instance."""
    return ConflictClassifier(llm_client=llm_client)