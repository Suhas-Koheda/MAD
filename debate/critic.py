"""
Critic agent for the debate module.
Identifies weaknesses in proposals and challenges reasoning.
"""
from typing import List, Dict, Any, Optional
from loguru import logger

from evidence.models import (
    Evidence, Claim, ConflictType, DebateArgument
)
from config import settings


class Critic:
    """
    Critiques proposals by identifying logical weaknesses,
    unsupported claims, and reasoning gaps.
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.demo_mode = settings.demo_mode
        self.call_count = 0
    
    def critique(
        self,
        query: str,
        proposal: DebateArgument,
        evidence_a: Evidence,
        evidence_b: Evidence,
        conflict_type: ConflictType,
        round_number: int,
    ) -> List[DebateArgument]:
        """
        Generate critiques of a proposal.
        
        Args:
            query: Original user query
            proposal: The proposal to critique
            evidence_a: Evidence from agent A
            evidence_b: Evidence from agent B
            conflict_type: Type of conflict detected
            round_number: Current debate round
            
        Returns:
            List of critique arguments
        """
        self.call_count += 1
        
        if self.demo_mode or not self.llm_client:
            critiques = self._demo_critique(
                query, proposal, evidence_a, evidence_b, conflict_type, round_number
            )
        else:
            critiques = self._llm_critique(
                query, proposal, evidence_a, evidence_b, conflict_type, round_number
            )
        
        return critiques
    
    def _demo_critique(
        self,
        query: str,
        proposal: DebateArgument,
        evidence_a: Evidence,
        evidence_b: Evidence,
        conflict_type: ConflictType,
        round_number: int,
    ) -> List[DebateArgument]:
        """Generate critiques in demo mode."""
        critiques = []
        
        claim_a = evidence_a.claims[0].text if evidence_a.claims else "No claims"
        claim_b = evidence_b.claims[0].text if evidence_b.claims else "No claims"
        source_a = evidence_a.sources[0].title if evidence_a.sources else "Unknown"
        source_b = evidence_b.sources[0].title if evidence_b.sources else "Unknown"
        
        # Critique 1: Challenge source completeness
        critique_1 = DebateArgument(
            round_number=round_number,
            role="critic",
            agent_id="critic_1",
            content=f"""CRITIQUE (Round {round_number}):

The proposal overlooks important source-specific factors:

1. Source Diversity: The analysis relies on limited sources ({source_a}, {source_b}).
   More diverse sources might reveal additional perspectives.

2. Source Authority: The proposal doesn't fully evaluate which source has higher authority.
   {source_a} may have different expertise than {source_b}.

3. Missing Context: The temporal information in source {source_a} may not account for
   recent developments that affect {source_b}'s position.

Recommendation: Strengthen the argument by explicitly comparing source credentials
and acknowledging temporal limitations.""",
            evidence_refs=[evidence_a.agent_id],
            conflict_type=conflict_type,
            strategy_used="source_evaluation",
        )
        critiques.append(critique_1)
        
        # Critique 2: Challenge logical consistency
        critique_2 = DebateArgument(
            round_number=round_number,
            role="critic",
            agent_id="critic_2",
            content=f"""CRITIQUE (Round {round_number}):

The proposal has logical gaps:

1. Unexamined Assumption: The argument assumes that recency automatically equals accuracy.
   Older sources might contain foundational information that remains valid.

2. Incomplete Resolution: The proposal doesn't fully resolve the conflict—it merely
   suggests which side to prefer. A complete answer should explain WHY.

3. Missing Evidence: The claims ({claim_a[:50]}...) vs ({claim_b[:50]}...)
   haven't been directly compared point-by-point.

Recommendation: Provide a direct, point-by-point comparison of the competing claims
with explicit reasoning for each comparison.""",
            evidence_refs=[evidence_b.agent_id],
            conflict_type=conflict_type,
            strategy_used="logical_analysis",
        )
        critiques.append(critique_2)
        
        return critiques
    
    def _llm_critique(
        self,
        query: str,
        proposal: DebateArgument,
        evidence_a: Evidence,
        evidence_b: Evidence,
        conflict_type: ConflictType,
        round_number: int,
    ) -> List[DebateArgument]:
        """Generate critiques using LLM."""
        try:
            claim_a = evidence_a.claims[0].text if evidence_a.claims else "No claims"
            claim_b = evidence_b.claims[0].text if evidence_b.claims else "No claims"
            source_a = evidence_a.sources[0].title if evidence_a.sources else "Unknown"
            source_b = evidence_b.sources[0].title if evidence_b.sources else "Unknown"
            
            prompt = f"""You are a critical analyst in a structured debate.

QUESTION: {query}
CONFLICT TYPE: {conflict_type.value}

ORIGINAL CLAIMS:
- {source_a}: {claim_a}
- {source_b}: {claim_b}

PROPOSAL TO CRITIQUE:
{proposal.content}

Generate TWO distinct critiques that:
1. Identify logical weaknesses or unsupported claims
2. Challenge assumptions the proposer made
3. Point out missing evidence or reasoning gaps
4. Suggest specific improvements

Format each critique clearly as CRITIQUE 1 and CRITIQUE 2:"""
            
            response = self.llm_client.chat(prompt)
            
            # Split into two critiques
            parts = response.split("CRITIQUE 2")
            if len(parts) == 2:
                content_1 = parts[0].replace("CRITIQUE 1", "").strip()
                content_2 = parts[1].strip()
            else:
                content_1 = response[:len(response)//2]
                content_2 = response[len(response)//2:]
            
            return [
                DebateArgument(
                    round_number=round_number,
                    role="critic",
                    agent_id="critic_1",
                    content=content_1,
                    evidence_refs=[evidence_a.agent_id],
                    conflict_type=conflict_type,
                ),
                DebateArgument(
                    round_number=round_number,
                    role="critic",
                    agent_id="critic_2",
                    content=content_2,
                    evidence_refs=[evidence_b.agent_id],
                    conflict_type=conflict_type,
                ),
            ]
            
        except Exception as e:
            logger.error(f"LLM critique generation failed: {e}")
            return self._demo_critique(
                query, proposal, evidence_a, evidence_b, conflict_type, round_number
            )