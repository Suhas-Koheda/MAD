"""
Final answer generator for producing structured responses.
"""
from typing import List, Dict, Any, Optional
from loguru import logger

from evidence.models import (
    Evidence, Conflict, ConflictType, DebateRound, 
    JudgeResult, FinalAnswer, Source
)
from config import settings


class AnswerGenerator:
    """
    Generates final answers from debate outcomes or direct evidence.
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.demo_mode = settings.demo_mode
    
    def generate(
        self,
        query: str,
        evidence_a: Evidence,
        evidence_b: Evidence,
        debate_triggered: bool = False,
        conflict: Optional[Conflict] = None,
        debate_rounds: Optional[List[DebateRound]] = None,
        judge_result: Optional[JudgeResult] = None,
    ) -> FinalAnswer:
        """
        Generate the final answer.
        
        Args:
            query: Original user query
            evidence_a: Evidence from agent A
            evidence_b: Evidence from agent B
            debate_triggered: Whether debate was triggered
            conflict: The detected conflict (if any)
            debate_rounds: List of debate rounds (if debate occurred)
            judge_result: Judge's evaluation (if debate occurred)
            
        Returns:
            FinalAnswer with complete response
        """
        if debate_triggered and judge_result:
            return self._generate_from_debate(
                query, evidence_a, evidence_b, conflict,
                debate_rounds, judge_result
            )
        else:
            return self._generate_direct(
                query, evidence_a, evidence_b
            )
    
    def _generate_direct(
        self,
        query: str,
        evidence_a: Evidence,
        evidence_b: Evidence,
    ) -> FinalAnswer:
        """Generate answer directly when evidence agrees."""
        # Combine evidence from both agents
        all_claims = evidence_a.claims + evidence_b.claims
        all_sources = evidence_a.sources + evidence_b.sources
        
        # Create unified answer
        answer_parts = []
        answer_parts.append(f"Based on evidence from {len(all_sources)} sources:")
        answer_parts.append("")
        
        # Add claims from both agents
        if evidence_a.claims:
            answer_parts.append(f"From {evidence_a.agent_id}:")
            for claim in evidence_a.claims[:3]:
                answer_parts.append(f"  • {claim.text}")
            answer_parts.append("")
        
        if evidence_b.claims:
            answer_parts.append(f"From {evidence_b.agent_id}:")
            for claim in evidence_b.claims[:3]:
                answer_parts.append(f"  • {claim.text}")
            answer_parts.append("")
        
        answer = "\n".join(answer_parts)
        
        # Create reasoning
        reasoning = (
            f"Evidence from both agents ({evidence_a.agent_id} and {evidence_b.agent_id}) "
            f"was consistent. No significant disagreement was detected. "
            f"The answer combines corroborating evidence from {len(all_sources)} sources."
        )
        
        # Deduplicate sources
        unique_sources = self._deduplicate_sources(all_sources)
        
        return FinalAnswer(
            query=query,
            answer=answer,
            reasoning=reasoning,
            evidence=[evidence_a, evidence_b],
            sources=unique_sources,
            debate_triggered=False,
            conflict=None,
            debate_rounds=[],
            judge_result=None,
            metadata={
                "evidence_agreement": True,
                "total_claims": len(all_claims),
                "total_sources": len(unique_sources),
            },
        )
    
    def _generate_from_debate(
        self,
        query: str,
        evidence_a: Evidence,
        evidence_b: Evidence,
        conflict: Optional[Conflict],
        debate_rounds: Optional[List[DebateRound]],
        judge_result: JudgeResult,
    ) -> FinalAnswer:
        """Generate answer from debate outcome."""
        # Select winning evidence based on judge
        if judge_result.winner == "candidate_1":
            winning_evidence = evidence_a
            losing_evidence = evidence_b
        else:
            winning_evidence = evidence_b
            losing_evidence = evidence_a
        
        # Build answer
        answer_parts = []
        answer_parts.append(f"Answer (resolved through evidence-based debate):")
        answer_parts.append("")
        
        # Get winner's answer
        if judge_result.winner == "candidate_1":
            answer_parts.append(evidence_a.answer)
        else:
            answer_parts.append(evidence_b.answer)
        
        answer_parts.append("")
        
        # Add conflict resolution
        if conflict:
            answer_parts.append(f"Conflict Detected: {conflict.type.value.upper()}")
            answer_parts.append(f"Resolution: {conflict.explanation}")
            answer_parts.append("")
        
        # Add judge's reasoning
        answer_parts.append(f"Judge's Reasoning: {judge_result.reason}")
        
        answer = "\n".join(answer_parts)
        
        # Create reasoning
        reasoning = (
            f"A {conflict.type.value if conflict else 'unknown'} conflict was detected "
            f"between evidence from {evidence_a.agent_id} and {evidence_b.agent_id}. "
            f"After {len(debate_rounds) if debate_rounds else 0} rounds of structured debate, "
            f"the judge selected {judge_result.winner} with "
            f"{judge_result.confidence:.0%} confidence."
        )
        
        # Collect all sources
        all_sources = evidence_a.sources + evidence_b.sources
        unique_sources = self._deduplicate_sources(all_sources)
        
        return FinalAnswer(
            query=query,
            answer=answer,
            reasoning=reasoning,
            evidence=[evidence_a, evidence_b],
            sources=unique_sources,
            debate_triggered=True,
            conflict=conflict,
            debate_rounds=debate_rounds or [],
            judge_result=judge_result,
            metadata={
                "evidence_agreement": False,
                "debate_rounds": len(debate_rounds) if debate_rounds else 0,
                "winner": judge_result.winner,
                "confidence": judge_result.confidence,
                "conflict_type": conflict.type.value if conflict else "unknown",
            },
        )
    
    def _deduplicate_sources(self, sources: List[Source]) -> List[Source]:
        """Remove duplicate sources by URL."""
        seen_urls = set()
        unique = []
        for source in sources:
            if source.url not in seen_urls:
                seen_urls.add(source.url)
                unique.append(source)
        return unique
    
    def format_for_display(self, answer: FinalAnswer) -> str:
        """Format answer for human-readable display."""
        lines = []
        lines.append("=" * 60)
        lines.append("MULTI-AGENT DEBATE SYSTEM - ANSWER")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Query: {answer.query}")
        lines.append("")
        
        if answer.debate_triggered:
            lines.append(f"Debate: YES ({len(answer.debate_rounds)} rounds)")
            if answer.conflict:
                lines.append(f"Conflict Type: {answer.conflict.type.value}")
        else:
            lines.append("Debate: NO (evidence agreed)")
        
        lines.append("")
        lines.append("-" * 60)
        lines.append(answer.answer)
        lines.append("-" * 60)
        lines.append("")
        
        if answer.reasoning:
            lines.append(f"Reasoning: {answer.reasoning}")
            lines.append("")
        
        if answer.sources:
            lines.append("Sources:")
            for i, source in enumerate(answer.sources, 1):
                lines.append(f"  {i}. {source.title}")
                lines.append(f"     {source.url}")
            lines.append("")
        
        if answer.judge_result:
            lines.append("Judge Evaluation:")
            lines.append(f"  Winner: {answer.judge_result.winner}")
            lines.append(f"  Confidence: {answer.judge_result.confidence:.0%}")
            lines.append(f"  Evidence Score: {answer.judge_result.evidence_score:.0%}")
        
        return "\n".join(lines)


def get_answer_generator(llm_client=None) -> AnswerGenerator:
    """Get or create an answer generator instance."""
    return AnswerGenerator(llm_client=llm_client)