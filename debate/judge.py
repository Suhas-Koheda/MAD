"""
Judge for evaluating debate outcomes and selecting the best answer.
"""
import json
import re
from typing import List, Dict, Any, Optional
from loguru import logger

from evidence.models import (
    Evidence, ConflictType, DebateArgument, DebateRound, JudgeResult
)
from config import settings


JUDGE_PROMPT = """You are a fair and impartial judge evaluating debate arguments.

You must evaluate based on EVIDENCE, not persuasion.

QUESTION: {query}

EVIDENCE:
Agent A ({agent_a_id}):
{evidence_a_text}

Agent B ({agent_b_id}):
{evidence_b_text}

DEBATE TRANSCRIPT:
{debate_transcript}

Evaluate the following criteria and provide scores:

1. EVIDENCE SUPPORT (0.0-1.0): How well does the answer support its claims with evidence?
2. SOURCE QUALITY (0.0-1.0): How authoritative and reliable are the sources used?
3. LOGICAL CONSISTENCY (0.0-1.0): Is the reasoning logically sound?
4. RELEVANCE (0.0-1.0): How well does the answer address the question?
5. CONFLICT RESOLUTION (0.0-1.0): How well does the answer resolve the underlying conflict?

Return ONLY a JSON object:
{{
    "winner": "candidate_1" or "candidate_2",
    "reason": "Brief explanation",
    "evidence_score": 0.0-1.0,
    "source_score": 0.0-1.0,
    "reasoning_score": 0.0-1.0,
    "relevance_score": 0.0-1.0,
    "conflict_resolution_score": 0.0-1.0,
    "confidence": 0.0-1.0,
    "winner_content": "The winning answer/argument"
}}
"""


class Judge:
    """
    Evaluates debate outcomes and selects the best candidate answer.
    Prioritizes evidence over persuasion.
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.demo_mode = settings.demo_mode
        self.call_count = 0
    
    def evaluate(
        self,
        query: str,
        evidence_a: Evidence,
        evidence_b: Evidence,
        debate_rounds: List[DebateRound],
        conflict_type: ConflictType,
    ) -> JudgeResult:
        """
        Evaluate the debate and select the best answer.
        
        Args:
            query: Original user query
            evidence_a: Evidence from agent A
            evidence_b: Evidence from agent B
            debate_rounds: List of completed debate rounds
            conflict_type: Type of conflict detected
            
        Returns:
            JudgeResult with winner and scores
        """
        self.call_count += 1
        
        if self.demo_mode or not self.llm_client:
            result = self._demo_evaluate(
                query, evidence_a, evidence_b, debate_rounds, conflict_type
            )
        else:
            result = self._llm_evaluate(
                query, evidence_a, evidence_b, debate_rounds, conflict_type
            )
        
        return result
    
    def _demo_evaluate(
        self,
        query: str,
        evidence_a: Evidence,
        evidence_b: Evidence,
        debate_rounds: List[DebateRound],
        conflict_type: ConflictType,
    ) -> JudgeResult:
        """Evaluate in demo mode."""
        # Simple heuristic evaluation based on evidence strength
        score_a = self._score_evidence(evidence_a)
        score_b = self._score_evidence(evidence_b)
        
        # Consider debate arguments
        if debate_rounds:
            last_round = debate_rounds[-1]
            if last_round.reviser_argument:
                # Check if revision addresses the conflict well
                if conflict_type.value in last_round.reviser_argument.content.lower():
                    # Bonus for addressing the conflict type
                    if score_a >= score_b:
                        score_a += 0.05
                    else:
                        score_b += 0.05
        
        winner = "candidate_1" if score_a >= score_b else "candidate_2"
        
        # Determine winner content
        if winner == "candidate_1":
            winner_content = evidence_a.answer
        else:
            winner_content = evidence_b.answer
        
        # Construct reasoning
        if conflict_type == ConflictType.TEMPORAL:
            reason = (
                f"The debate analyzed temporal aspects of the conflict. "
                f"Candidate {winner[-1]} provided more current and relevant information "
                f"while acknowledging the temporal context of both claims."
            )
        elif conflict_type == ConflictType.FACTUAL:
            reason = (
                f"The debate compared factual claims directly. "
                f"Candidate {winner[-1]} had stronger evidentiary support and "
                f"more authoritative sourcing."
            )
        elif conflict_type == ConflictType.VERSION:
            reason = (
                f"The debate identified version differences. "
                f"Candidate {winner[-1]} correctly identified the relevant version "
                f"and provided version-specific information."
            )
        elif conflict_type == ConflictType.CONTEXTUAL:
            reason = (
                f"The debate analyzed contextual differences. "
                f"Candidate {winner[-1]} showed better understanding of when "
                f"each claim applies."
            )
        else:
            reason = (
                f"The debate evaluated source reliability. "
                f"Candidate {winner[-1]} demonstrated stronger source authority "
                f"and credibility."
            )
        
        return JudgeResult(
            winner=winner,
            reason=reason,
            evidence_score=max(score_a, score_b),
            reasoning_score=max(score_a, score_b) - 0.05,
            source_score=max(score_a, score_b) - 0.03,
            confidence=max(score_a, score_b) - 0.02,
            scores={
                "candidate_1": score_a,
                "candidate_2": score_b,
            },
        )
    
    def _score_evidence(self, evidence: Evidence) -> float:
        """Score evidence based on claims and sources."""
        score = 0.5  # Base score
        
        # Add points for having claims
        if evidence.claims:
            score += min(0.2, len(evidence.claims) * 0.02)
        
        # Add points for having sources
        if evidence.sources:
            score += min(0.2, len(evidence.sources) * 0.04)
        
        # Add points for longer, more detailed answers
        if len(evidence.answer) > 100:
            score += 0.1
        
        return min(0.95, score)
    
    def _llm_evaluate(
        self,
        query: str,
        evidence_a: Evidence,
        evidence_b: Evidence,
        debate_rounds: List[DebateRound],
        conflict_type: ConflictType,
    ) -> JudgeResult:
        """Evaluate using LLM."""
        try:
            # Build debate transcript
            transcript = self._build_transcript(debate_rounds)
            
            evidence_a_text = self._format_evidence(evidence_a)
            evidence_b_text = self._format_evidence(evidence_b)
            
            prompt = JUDGE_PROMPT.format(
                query=query,
                agent_a_id=evidence_a.agent_id,
                agent_b_id=evidence_b.agent_id,
                evidence_a_text=evidence_a_text,
                evidence_b_text=evidence_b_text,
                debate_transcript=transcript,
            )
            
            response = self.llm_client.chat(prompt)
            
            # Parse JSON response
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                winner = result.get("winner", "candidate_1")
                if winner not in ["candidate_1", "candidate_2"]:
                    winner = "candidate_1"
                
                return JudgeResult(
                    winner=winner,
                    reason=result.get("reason", "LLM evaluation"),
                    evidence_score=float(result.get("evidence_score", 0.8)),
                    reasoning_score=float(result.get("reasoning_score", 0.8)),
                    source_score=float(result.get("source_score", 0.8)),
                    confidence=float(result.get("confidence", 0.8)),
                    scores={
                        "candidate_1": float(result.get("evidence_score", 0.8)),
                        "candidate_2": float(result.get("reasoning_score", 0.75)),
                    },
                )
            
            # Fallback to demo evaluation
            return self._demo_evaluate(
                query, evidence_a, evidence_b, debate_rounds, conflict_type
            )
            
        except Exception as e:
            logger.error(f"LLM judge evaluation failed: {e}")
            return self._demo_evaluate(
                query, evidence_a, evidence_b, debate_rounds, conflict_type
            )
    
    def _build_transcript(self, rounds: List[DebateRound]) -> str:
        """Build debate transcript from rounds."""
        if not rounds:
            return "No debate rounds completed."
        
        lines = []
        for round in rounds:
            lines.append(f"\n=== ROUND {round.round_number} ===\n")
            
            if round.proposer_argument:
                lines.append(f"PROPOSER:\n{round.proposer_argument.content}\n")
            
            for i, critic in enumerate(round.critic_arguments):
                lines.append(f"CRITIC {i+1}:\n{critic.content}\n")
            
            if round.reviser_argument:
                lines.append(f"REVISER:\n{round.reviser_argument.content}\n")
        
        return "\n".join(lines)
    
    def _format_evidence(self, evidence: Evidence) -> str:
        """Format evidence for prompt."""
        parts = [
            f"Query: {evidence.query}",
            f"Answer: {evidence.answer}",
        ]
        
        if evidence.claims:
            parts.append("Claims:")
            for claim in evidence.claims[:5]:
                parts.append(f"  - {claim.text}")
        
        if evidence.sources:
            parts.append("Sources:")
            for source in evidence.sources[:5]:
                parts.append(f"  - {source.title}: {source.url}")
        
        return "\n".join(parts)


def get_judge(llm_client=None) -> Judge:
    """Get or create a judge instance."""
    return Judge(llm_client=llm_client)