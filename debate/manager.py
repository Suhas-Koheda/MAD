"""
Debate manager for orchestrating the debate process.
Manages the Propose → Critique → Revise cycle.
"""
from typing import List, Dict, Any, Optional
from loguru import logger

from evidence.models import (
    Evidence, ConflictType, Conflict, DebateArgument, DebateRound
)
from debate.proposer import Proposer
from debate.critic import Critic
from debate.reviser import Reviser
from config import settings


class DebateManager:
    """
    Manages the debate process between multiple agents.
    Runs 2-3 rounds of Propose → Critique → Revise.
    """
    
    def __init__(self, llm_client=None, max_rounds: Optional[int] = None):
        self.llm_client = llm_client
        self.max_rounds = max_rounds or settings.max_debate_rounds
        self.proposer = Proposer(llm_client)
        self.critic = Critic(llm_client)
        self.reviser = Reviser(llm_client)
        self.demo_mode = settings.demo_mode
        self.total_llm_calls = 0
    
    def run_debate(
        self,
        query: str,
        evidence_a: Evidence,
        evidence_b: Evidence,
        conflict: Conflict,
    ) -> List[DebateRound]:
        """
        Run the full debate process.
        
        Args:
            query: Original user query
            evidence_a: Evidence from agent A
            evidence_b: Evidence from agent B
            conflict: The detected conflict
            
        Returns:
            List of completed debate rounds
        """
        logger.info(
            f"Starting debate for conflict type: {conflict.type.value}, "
            f"max rounds: {self.max_rounds}"
        )
        
        rounds = []
        current_proposal = None
        
        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"Debate Round {round_num}/{self.max_rounds}")
            
            # PROPOSE
            proposal = self.proposer.propose(
                query=query,
                evidence_a=evidence_a,
                evidence_b=evidence_b,
                conflict_type=conflict.type,
                conflict_explanation=conflict.explanation,
                round_number=round_num,
            )
            self.total_llm_calls += 1
            
            # CRITIQUE
            critiques = self.critic.critique(
                query=query,
                proposal=proposal,
                evidence_a=evidence_a,
                evidence_b=evidence_b,
                conflict_type=conflict.type,
                round_number=round_num,
            )
            self.total_llm_calls += len(critiques)
            
            # REVISE
            revision = self.reviser.revise(
                query=query,
                proposal=proposal,
                critiques=critiques,
                evidence_a=evidence_a,
                evidence_b=evidence_b,
                conflict_type=conflict.type,
                round_number=round_num,
            )
            self.total_llm_calls += 1
            
            # Create debate round
            debate_round = DebateRound(
                round_number=round_num,
                proposer_argument=proposal,
                critic_arguments=critiques,
                reviser_argument=revision,
            )
            rounds.append(debate_round)
            
            current_proposal = revision
            logger.info(f"Round {round_num} completed")
        
        logger.info(
            f"Debate completed: {len(rounds)} rounds, "
            f"total LLM calls: {self.total_llm_calls}"
        )
        
        return rounds
    
    def get_debate_summary(self, rounds: List[DebateRound]) -> Dict[str, Any]:
        """Get summary of the debate process."""
        if not rounds:
            return {"rounds": 0, "status": "no_debate"}
        
        last_round = rounds[-1]
        return {
            "rounds": len(rounds),
            "final_proposal": last_round.reviser_argument.content if last_round.reviser_argument else "",
            "total_proposals": len([r for r in rounds if r.proposer_argument]),
            "total_critiques": sum(len(r.critic_arguments) for r in rounds),
            "total_revisions": len([r for r in rounds if r.reviser_argument]),
            "conflict_type": rounds[0].proposer_argument.conflict_type.value if rounds[0].proposer_argument else "unknown",
        }


def get_debate_manager(llm_client=None, max_rounds: Optional[int] = None) -> DebateManager:
    """Get or create a debate manager instance."""
    return DebateManager(llm_client=llm_client, max_rounds=max_rounds)