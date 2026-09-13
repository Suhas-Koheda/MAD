"""
Evidence package for the Multi-Agent Debate framework.
"""
from .models import (
    ConflictType,
    NLILabel,
    Source,
    Claim,
    Evidence,
    EvidenceComparison,
    Conflict,
    DebateArgument,
    DebateRound,
    JudgeResult,
    FinalAnswer,
    ExecutionTrace,
)
from .comparator import EvidenceComparator, find_most_contradictory_pair
from .disagreement_detector import NLIDetector, MockNLIDetector, get_nli_detector

__all__ = [
    "ConflictType",
    "NLILabel",
    "Source",
    "Claim",
    "Evidence",
    "EvidenceComparison",
    "Conflict",
    "DebateArgument",
    "DebateRound",
    "JudgeResult",
    "FinalAnswer",
    "ExecutionTrace",
    "EvidenceComparator",
    "find_most_contradictory_pair",
    "NLIDetector",
    "MockNLIDetector",
    "get_nli_detector",
]