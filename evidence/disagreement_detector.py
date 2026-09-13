"""
NLI-based disagreement detector using Hugging Face transformers.
"""
import os
from typing import Dict, Any, Optional, List
from loguru import logger
from functools import lru_cache

from evidence.models import NLILabel
from config import settings


class NLIDetector:
    """
    Natural Language Inference detector for identifying contradictions
    between evidence claims.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        threshold: Optional[float] = None,
    ):
        self.model_name = model_name or settings.nli_model
        self.device = device or settings.nli_device
        self.threshold = threshold or settings.disagreement_threshold
        self._model = None
        self._tokenizer = None
        self._pipeline = None
    
    def _load_model(self):
        """Load the NLI model and tokenizer."""
        if self._pipeline is not None:
            return
        
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            logger.info(f"Loading NLI model: {self.model_name} on {self.device}")
            
            # Determine device
            if self.device == "auto":
                device = 0 if torch.cuda.is_available() else -1
            elif self.device == "cuda":
                device = 0 if torch.cuda.is_available() else -1
            else:
                device = -1  # CPU
            
            # Load tokenizer and model
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Create pipeline
            self._pipeline = pipeline(
                "text-classification",
                model=model,
                tokenizer=self._tokenizer,
                device=device,
                return_all_scores=True,
                function_to_apply="softmax",
            )
            
            logger.info("NLI model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load NLI model: {e}")
            raise
    
    def predict(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """
        Predict NLI relationship between premise and hypothesis.
        
        Args:
            premise: First text (premise)
            hypothesis: Second text (hypothesis)
            
        Returns:
            Dictionary with label, score, and all_scores
        """
        self._load_model()
        
        # Format input for NLI
        input_text = f"{premise} </s> {hypothesis}"
        
        try:
            results = self._pipeline(input_text)
            
            # Process results - pipeline returns list of lists with all scores
            all_scores = {}
            for item in results[0]:
                label = item["label"].lower()
                # Map labels to standard NLI labels
                if "entail" in label:
                    all_scores[NLILabel.ENTAILMENT] = item["score"]
                elif "contradict" in label:
                    all_scores[NLILabel.CONTRADICTION] = item["score"]
                elif "neutral" in label:
                    all_scores[NLILabel.NEUTRAL] = item["score"]
                else:
                    all_scores[label] = item["score"]
            
            # Find highest scoring label
            if not all_scores:
                return {
                    "label": NLILabel.NEUTRAL,
                    "score": 0.0,
                    "all_scores": {NLILabel.NEUTRAL: 1.0},
                }
            
            best_label = max(all_scores, key=all_scores.get)
            best_score = all_scores[best_label]
            
            return {
                "label": best_label,
                "score": best_score,
                "all_scores": all_scores,
            }
            
        except Exception as e:
            logger.error(f"NLI prediction failed: {e}")
            return {
                "label": NLILabel.NEUTRAL,
                "score": 0.0,
                "all_scores": {NLILabel.NEUTRAL: 1.0},
            }
    
    def predict_batch(self, pairs: List[tuple]) -> List[Dict[str, Any]]:
        """
        Predict NLI for multiple premise-hypothesis pairs.
        
        Args:
            pairs: List of (premise, hypothesis) tuples
            
        Returns:
            List of prediction dictionaries
        """
        self._load_model()
        
        inputs = [f"{p} </s> {h}" for p, h in pairs]
        
        try:
            results = self._pipeline(inputs)
            
            processed = []
            for result in results:
                all_scores = {}
                for item in result:
                    label = item["label"].lower()
                    if "entail" in label:
                        all_scores[NLILabel.ENTAILMENT] = item["score"]
                    elif "contradict" in label:
                        all_scores[NLILabel.CONTRADICTION] = item["score"]
                    elif "neutral" in label:
                        all_scores[NLILabel.NEUTRAL] = item["score"]
                    else:
                        all_scores[label] = item["score"]
                
                if not all_scores:
                    processed.append({
                        "label": NLILabel.NEUTRAL,
                        "score": 0.0,
                        "all_scores": {NLILabel.NEUTRAL: 1.0},
                    })
                else:
                    best_label = max(all_scores, key=all_scores.get)
                    best_score = all_scores[best_label]
                    processed.append({
                        "label": best_label,
                        "score": best_score,
                        "all_scores": all_scores,
                    })
            
            return processed
            
        except Exception as e:
            logger.error(f"NLI batch prediction failed: {e}")
            return [
                {"label": NLILabel.NEUTRAL, "score": 0.0, "all_scores": {NLILabel.NEUTRAL: 1.0}}
                for _ in pairs
            ]
    
    def detect_disagreement(self, text_a: str, text_b: str) -> Dict[str, Any]:
        """
        Detect if two texts disagree.
        
        Args:
            text_a: First text
            text_b: Second text
            
        Returns:
            Dictionary with disagreement detection result
        """
        result = self.predict(text_a, text_b)
        
        is_contradiction = result["label"] == NLILabel.CONTRADICTION
        contradiction_score = result["all_scores"].get(NLILabel.CONTRADICTION, 0.0)
        
        return {
            "disagreement": is_contradiction and contradiction_score >= self.threshold,
            "label": result["label"],
            "contradiction_score": contradiction_score,
            "entailment_score": result["all_scores"].get(NLILabel.ENTAILMENT, 0.0),
            "neutral_score": result["all_scores"].get(NLILabel.NEUTRAL, 0.0),
            "threshold": self.threshold,
            "all_scores": result["all_scores"],
        }


class MockNLIDetector:
    """Mock NLI detector for demo/testing without model loading."""
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        # Predefined contradictions for demo
        self.contradiction_patterns = [
            (["2019", "2020"], 0.91),
            (["2022", "2024"], 0.89),
            (["enterprise", "individual"], 0.85),
            (["version 1", "version 2"], 0.88),
            (["deprecated", "current"], 0.92),
            (["adults", "children"], 0.86),
            (["secure", "insecure"], 0.87),
            (["increased", "decreased"], 0.94),
        ]
    
    def predict(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """Mock prediction based on keyword patterns."""
        premise_lower = premise.lower()
        hypothesis_lower = hypothesis.lower()
        
        # Check for known contradiction patterns
        for keywords, score in self.contradiction_patterns:
            kw1, kw2 = keywords
            if (kw1 in premise_lower and kw2 in hypothesis_lower) or \
               (kw2 in premise_lower and kw1 in hypothesis_lower):
                return {
                    "label": NLILabel.CONTRADICTION,
                    "score": score,
                    "all_scores": {
                        NLILabel.CONTRADICTION: score,
                        NLILabel.ENTAILMENT: 0.05,
                        NLILabel.NEUTRAL: 0.05,
                    },
                }
        
        # Check for entailment (similar content)
        common_words = set(premise_lower.split()) & set(hypothesis_lower.split())
        if len(common_words) > 3:
            return {
                "label": NLILabel.ENTAILMENT,
                "score": 0.85,
                "all_scores": {
                    NLILabel.ENTAILMENT: 0.85,
                    NLILabel.CONTRADICTION: 0.05,
                    NLILabel.NEUTRAL: 0.10,
                },
            }
        
        # Default to neutral
        return {
            "label": NLILabel.NEUTRAL,
            "score": 0.60,
            "all_scores": {
                NLILabel.NEUTRAL: 0.60,
                NLILabel.ENTAILMENT: 0.20,
                NLILabel.CONTRADICTION: 0.20,
            },
        }
    
    def detect_disagreement(self, text_a: str, text_b: str) -> Dict[str, Any]:
        """Mock disagreement detection."""
        result = self.predict(text_a, text_b)
        
        is_contradiction = result["label"] == NLILabel.CONTRADICTION
        contradiction_score = result["all_scores"].get(NLILabel.CONTRADICTION, 0.0)
        
        return {
            "disagreement": is_contradiction and contradiction_score >= self.threshold,
            "label": result["label"],
            "contradiction_score": contradiction_score,
            "entailment_score": result["all_scores"].get(NLILabel.ENTAILMENT, 0.0),
            "neutral_score": result["all_scores"].get(NLILabel.NEUTRAL, 0.0),
            "threshold": self.threshold,
            "all_scores": result["all_scores"],
        }


@lru_cache()
def get_nli_detector() -> NLIDetector:
    """Get or create NLI detector instance."""
    if settings.demo_mode:
        logger.info("Using MockNLIDetector (DEMO MODE)")
        return MockNLIDetector(threshold=settings.disagreement_threshold)
    
    logger.info("Using real NLIDetector")
    return NLIDetector(
        model_name=settings.nli_model,
        device=settings.nli_device,
        threshold=settings.disagreement_threshold,
    )