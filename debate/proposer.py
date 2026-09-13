"""
Proposer agent for the debate module.
Generates initial arguments based on evidence and conflict type.
"""
from typing import List, Dict, Any, Optional
from loguru import logger

from evidence.models import (
    Evidence, Claim, ConflictType, DebateArgument
)
from config import settings


# Conflict-specific strategies for the proposer
CONFLICT_STRATEGIES = {
    ConflictType.FACTUAL: """You are arguing about a factual disagreement. 
Focus on direct evidence. Compare the competing factual claims directly against the retrieved evidence.
Identify which claim has stronger direct support. Cite specific sources.""",
    
    ConflictType.TEMPORAL: """You are arguing about a temporal disagreement.
Construct a timeline. Determine when each claim was true.
Consider if both claims might be true at different times.
Determine which claim best answers the user's current question.""",
    
    ConflictType.VERSION: """You are arguing about a version disagreement.
Identify the versions involved. Determine which version is relevant to the question.
Resolve contradictions within the correct version context.""",
    
    ConflictType.CONTEXTUAL: """You are arguing about a contextual disagreement.
Identify the conditions under which each claim holds.
Determine whether the apparent contradiction disappears when context is considered.""",
    
    ConflictType.SOURCE: """You are arguing about a source disagreement.
Evaluate source authority, provenance, specificity, and reliability.
Prefer stronger primary evidence where appropriate.""",
}


class Proposer:
    """
    Generates initial arguments in the debate.
    Uses conflict-type-specific strategies.
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.demo_mode = settings.demo_mode
        self.call_count = 0
    
    def propose(
        self,
        query: str,
        evidence_a: Evidence,
        evidence_b: Evidence,
        conflict_type: ConflictType,
        conflict_explanation: str,
        round_number: int = 1,
    ) -> DebateArgument:
        """
        Generate a proposal argument.
        
        Args:
            query: Original user query
            evidence_a: Evidence from agent A
            evidence_b: Evidence from agent B
            conflict_type: Type of conflict detected
            conflict_explanation: Explanation of the conflict
            round_number: Current debate round
            
        Returns:
            DebateArgument with the proposal
        """
        self.call_count += 1
        
        strategy = CONFLICT_STRATEGIES.get(
            conflict_type, 
            CONFLICT_STRATEGIES[ConflictType.FACTUAL]
        )
        
        if self.demo_mode or not self.llm_client:
            argument = self._demo_propose(
                query, evidence_a, evidence_b, conflict_type, round_number
            )
        else:
            argument = self._llm_propose(
                query, evidence_a, evidence_b, conflict_type,
                conflict_explanation, strategy, round_number
            )
        
        return argument
    
    def _demo_propose(
        self,
        query: str,
        evidence_a: Evidence,
        evidence_b: Evidence,
        conflict_type: ConflictType,
        round_number: int,
    ) -> DebateArgument:
        """Generate proposal in demo mode."""
        claim_a_text = evidence_a.claims[0].text if evidence_a.claims else "No claims"
        claim_b_text = evidence_b.claims[0].text if evidence_b.claims else "No claims"
        source_a = evidence_a.sources[0].title if evidence_a.sources else "Unknown"
        source_b = evidence_b.sources[0].title if evidence_b.sources else "Unknown"
        
        if conflict_type == ConflictType.TEMPORAL:
            content = f"""ROUND {round_number} PROPOSAL:

The question "{query}" has temporal conflict between:
- {claim_a_text} (from {source_a})
- {claim_b_text} (from {source_b})

Analysis: These claims refer to different time periods. Source A describes the situation 
from an earlier period, while Source B reflects more recent changes.

Recommendation: For current relevance, the more recent information should be weighted higher,
but both perspectives provide valuable context.

Strategy: Construct timeline and determine temporal relevance."""
        
        elif conflict_type == ConflictType.FACTUAL:
            content = f"""ROUND {round_number} PROPOSAL:

The question "{query}" has factual conflict between:
- {claim_a_text} (from {source_a})
- {claim_b_text} (from {source_b})

Analysis: These claims make mutually incompatible factual assertions.
Evidence from multiple sources must be compared directly.

Recommendation: Evaluate the strength of evidence supporting each claim.
Consider which source has stronger authority and more specific information.

Strategy: Direct evidence comparison and authority evaluation."""
        
        elif conflict_type == ConflictType.VERSION:
            content = f"""ROUND {round_number} PROPOSAL:

The question "{query}" has version-related conflict between:
- {claim_a_text} (from {source_a})
- {claim_b_text} (from {source_b})

Analysis: These claims refer to different versions or releases.
The contradiction may be valid across different versions.

Recommendation: Identify which version is most relevant to the question.
Resolve within the context of that specific version.

Strategy: Version identification and context-specific resolution."""
        
        elif conflict_type == ConflictType.CONTEXTUAL:
            content = f"""ROUND {round_number} PROPOSAL:

The question "{query}" has contextual conflict between:
- {claim_a_text} (from {source_a})
- {claim_b_text} (from {source_b})

Analysis: These claims apply under different conditions or contexts.
The apparent contradiction may dissolve when context is considered.

Recommendation: Identify the conditions under which each claim holds true.
Determine if both can be valid in their respective contexts.

Strategy: Context identification and conditional truth evaluation."""
        
        else:  # SOURCE or UNKNOWN
            content = f"""ROUND {round_number} PROPOSAL:

The question "{query}" has source-related conflict between:
- {claim_a_text} (from {source_a})
- {claim_b_text} (from {source_b})

Analysis: The disagreement stems from differences in source quality or reliability.
Source characteristics must be evaluated.

Recommendation: Evaluate source authority, recency, and specificity.
Prefer primary and authoritative sources.

Strategy: Source authority and reliability evaluation."""
        
        return DebateArgument(
            round_number=round_number,
            role="proposer",
            agent_id="proposer",
            content=content,
            evidence_refs=[evidence_a.agent_id, evidence_b.agent_id],
            conflict_type=conflict_type,
            strategy_used=conflict_type.value,
        )
    
    def _llm_propose(
        self,
        query: str,
        evidence_a: Evidence,
        evidence_b: Evidence,
        conflict_type: ConflictType,
        conflict_explanation: str,
        strategy: str,
        round_number: int,
    ) -> DebateArgument:
        """Generate proposal using LLM."""
        try:
            claim_a = evidence_a.claims[0].text if evidence_a.claims else "No claims available"
            claim_b = evidence_b.claims[0].text if evidence_b.claims else "No claims available"
            source_a = evidence_a.sources[0].title if evidence_a.sources else "Unknown"
            source_b = evidence_b.sources[0].title if evidence_b.sources else "Unknown"
            
            prompt = f"""{strategy}

QUESTION: {query}

CONFLICT DETECTED:
- Claim A ({source_a}): {claim_a}
- Claim B ({source_b}): {claim_b}

Conflict Type: {conflict_type.value}
Explanation: {conflict_explanation}

Generate a structured proposal (Round {round_number}) that:
1. Analyzes the evidence from both sides
2. Identifies the core issue in the conflict
3. Proposes an initial position supported by evidence
4. References specific sources

Provide your proposal:"""
            
            response = self.llm_client.chat(prompt)
            
            return DebateArgument(
                round_number=round_number,
                role="proposer",
                agent_id="proposer",
                content=response,
                evidence_refs=[evidence_a.agent_id, evidence_b.agent_id],
                conflict_type=conflict_type,
                strategy_used=conflict_type.value,
            )
            
        except Exception as e:
            logger.error(f"LLM proposal generation failed: {e}")
            return self._demo_propose(
                query, evidence_a, evidence_b, conflict_type, round_number
            )