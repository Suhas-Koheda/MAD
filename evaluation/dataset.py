"""
Evaluation dataset for the Multi-Agent Debate framework.
Contains questions with potentially conflicting evidence.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from loguru import logger

from evidence.models import ConflictType


class EvaluationSample(BaseModel):
    """A single evaluation sample."""
    id: str
    query: str
    expected_conflict: bool
    expected_conflict_type: Optional[str] = None
    expected_answer: Optional[str] = None
    category: str = "general"
    difficulty: str = "medium"
    metadata: Dict[str, Any] = {}


class EvaluationDataset:
    """
    Dataset of questions with potentially conflicting evidence
    for evaluating the MAD system.
    """
    
    def __init__(self, samples: Optional[List[EvaluationSample]] = None):
        self.samples = samples or self._get_default_samples()
    
    def _get_default_samples(self) -> List[EvaluationSample]:
        """Get default evaluation samples."""
        return [
            # TEMPORAL CONFLICTS
            EvaluationSample(
                id="temporal_1",
                query="When was the first iPhone released?",
                expected_conflict=True,
                expected_conflict_type="temporal",
                expected_answer="The first iPhone was announced in January 2007 and released in June 2007.",
                category="temporal",
                difficulty="easy",
            ),
            EvaluationSample(
                id="temporal_2",
                query="What is the current unemployment rate?",
                expected_conflict=True,
                expected_conflict_type="temporal",
                expected_answer="The unemployment rate changes monthly; check the most recent BLS report.",
                category="temporal",
                difficulty="medium",
            ),
            EvaluationSample(
                id="temporal_3",
                query="When did COVID-19 start?",
                expected_conflict=True,
                expected_conflict_type="temporal",
                expected_answer="COVID-19 was first identified in late 2019, with the WHO declaring it a pandemic in March 2020.",
                category="temporal",
                difficulty="medium",
            ),
            
            # FACTUAL CONFLICTS
            EvaluationSample(
                id="factual_1",
                query="What is the speed of light?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="The speed of light in vacuum is approximately 299,792,458 meters per second.",
                category="factual",
                difficulty="easy",
            ),
            EvaluationSample(
                id="factual_2",
                query="How many countries are in the European Union?",
                expected_conflict=True,
                expected_conflict_type="factual",
                expected_answer="The EU has 27 member states after Brexit.",
                category="factual",
                difficulty="medium",
            ),
            EvaluationSample(
                id="factual_3",
                query="What is the tallest building in the world?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="The Burj Khalifa in Dubai is the tallest building at 828 meters.",
                category="factual",
                difficulty="easy",
            ),
            
            # VERSION CONFLICTS
            EvaluationSample(
                id="version_1",
                query="What features does Python 3.10 support?",
                expected_conflict=True,
                expected_conflict_type="version",
                expected_answer="Python 3.10 introduced structural pattern matching and other features.",
                category="version",
                difficulty="medium",
            ),
            EvaluationSample(
                id="version_2",
                query="What changed in React 18?",
                expected_conflict=True,
                expected_conflict_type="version",
                expected_answer="React 18 introduced concurrent features and automatic batching.",
                category="version",
                difficulty="medium",
            ),
            
            # CONTEXTUAL CONFLICTS
            EvaluationSample(
                id="contextual_1",
                query="Is coffee healthy?",
                expected_conflict=True,
                expected_conflict_type="contextual",
                expected_answer="Coffee can be healthy in moderation but may have negative effects for some individuals.",
                category="contextual",
                difficulty="medium",
            ),
            EvaluationSample(
                id="contextual_2",
                query="Should I invest in stocks?",
                expected_conflict=True,
                expected_conflict_type="contextual",
                expected_answer="Investment decisions depend on individual risk tolerance, goals, and financial situation.",
                category="contextual",
                difficulty="hard",
            ),
            
            # SOURCE CONFLICTS
            EvaluationSample(
                id="source_1",
                query="Is email encryption really secure?",
                expected_conflict=True,
                expected_conflict_type="source",
                expected_answer="Email encryption security depends on implementation and proper key management.",
                category="source",
                difficulty="hard",
            ),
            EvaluationSample(
                id="source_2",
                query="Do supplements actually work?",
                expected_conflict=True,
                expected_conflict_type="source",
                expected_answer="Effectiveness varies by supplement and individual; peer-reviewed studies should be prioritized.",
                category="source",
                difficulty="hard",
            ),
            
            # NO CONFLICT (AGREEMENT)
            EvaluationSample(
                id="agree_1",
                query="What is the boiling point of water at sea level?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="Water boils at 100°C (212°F) at sea level.",
                category="agreement",
                difficulty="easy",
            ),
            EvaluationSample(
                id="agree_2",
                query="Who wrote Romeo and Juliet?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="William Shakespeare wrote Romeo and Juliet.",
                category="agreement",
                difficulty="easy",
            ),
            EvaluationSample(
                id="agree_3",
                query="What planet is closest to the Sun?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="Mercury is the planet closest to the Sun.",
                category="agreement",
                difficulty="easy",
            ),
        ]
    
    def load_from_file(self, filepath: str) -> None:
        """Load samples from JSON file."""
        path = Path(filepath)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                self.samples = [EvaluationSample(**item) for item in data]
            logger.info(f"Loaded {len(self.samples)} samples from {filepath}")
    
    def save_to_file(self, filepath: str) -> None:
        """Save samples to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [sample.model_dump() for sample in self.samples]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self.samples)} samples to {filepath}")
    
    def get_samples_by_category(self, category: str) -> List[EvaluationSample]:
        """Get samples filtered by category."""
        return [s for s in self.samples if s.category == category]
    
    def get_samples_by_conflict(self, has_conflict: bool) -> List[EvaluationSample]:
        """Get samples filtered by expected conflict."""
        return [s for s in self.samples if s.expected_conflict == has_conflict]
    
    def get_categories(self) -> List[str]:
        """Get all unique categories."""
        return list(set(s.category for s in self.samples))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        categories = {}
        for sample in self.samples:
            if sample.category not in categories:
                categories[sample.category] = {"total": 0, "conflicts": 0, "no_conflicts": 0}
            categories[sample.category]["total"] += 1
            if sample.expected_conflict:
                categories[sample.category]["conflicts"] += 1
            else:
                categories[sample.category]["no_conflicts"] += 1
        
        return {
            "total_samples": len(self.samples),
            "conflict_samples": len([s for s in self.samples if s.expected_conflict]),
            "no_conflict_samples": len([s for s in self.samples if not s.expected_conflict]),
            "categories": categories,
        }


def get_evaluation_dataset() -> EvaluationDataset:
    """Get or create evaluation dataset."""
    dataset = EvaluationDataset()
    
    # Try to load from file
    data_path = Path("data/conflicts.json")
    if data_path.exists():
        dataset.load_from_file(str(data_path))
    else:
        # Save default dataset
        dataset.save_to_file(str(data_path))
    
    return dataset