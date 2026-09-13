"""
Reviser agent for the debate module.
Revises arguments based on critique feedback.
"""
from typing import List, Dict, Any, Optional
from loguru import logger

from evidence.models import (
    Evidence, Claim, ConflictType, DebateArgument
)
from config import settings


class Reviser:
    """
    Revises arguments by incorporating critique feedback
    and producing improved reasoning.
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.demo_mode = settings.demo_mode
        self.call_count = 0
    
    def revise(
        self,
        query: str,
        proposal: DebateArgument,
        critiques: List[DebateArgument],
        evidence_a: Evidence,
        evidence_b: Evidence,
        conflict_type: ConflictType,
        round_number: int,
    ) -> DebateArgument:
        """
        Generate a revised argument incorporating critiques.
        
        Args:
            query: Original user query
            proposal: The original proposal
            critiques: List of critiques to address
            evidence_a: Evidence from agent A
            evidence_b: Evidence from agent B
            conflict_type: Type of conflict detected
            round_number: Current debate round
            
        Returns:
            Revised DebateArgument
        """
        self.call_count += 1
        
        if self.demo_mode or not self.llm_client:
            argument = self._demo_revise(
                query, proposal, critiques, evidence_a, evidence_b,
                conflict_type, round_number
            )
        else:
            argument = self._llm_revise(
                query, proposal, critiques, evidence_a, evidence_b,
                conflict_type, round_number
            )
        
        return argument
    
    def _demo_revise(
        self,
        query: str,
        proposal: DebateArgument,
        critiques: List[DebateArgument],
        evidence_a: Evidence,
        evidence_b: Evidence,
        conflict_type: ConflictType,
        round_number: int,
    ) -> DebateArgument:
        """Generate revision in demo mode."""
        claim_a = evidence_a.claims[0].text if evidence_a.claims else "No claims"
        claim_b = evidence_b.claims[0].text if evidence_b.claims else "No claims"
        source_a = evidence_a.sources[0].title if evidence_a.sources else "Unknown"
        source_b = evidence_b.sources[0].title if evidence_b.sources else "Unknown"
        
        critique_summary = "\n".join([
            f"- {c.content[:100]}..." for c in critiques
        ])
        
        if conflict_type == ConflictType.TEMPORAL:
            content = f"""ROUND {round_number} REVISED PROPOSAL:

Addressing critiques: Source diversity and temporal relevance.

Revised Analysis for "{query}":

1. TEMPORAL CONTEXT:
   - {source_a} reflects information valid as of earlier timeframe
   - {source_b} reflects more recent developments
   - Both can be valid within their respective time periods

2. EVIDENCE COMPARISON:
   - Claim A: {claim_a[:80]}...
   - Claim B: {claim_b[:80]}...
   - The key difference is temporal: conditions have changed over time

3. REVISED RECOMMENDATION:
   For answering the user's current question, the most recent information
   takes precedence. However, the historical context provides valuable background.
   
   Final Position: Current answer based on {source_b}, with {source_a}
   as historical context.

4. Addressing Critiques:
   - Acknowledged source diversity limitation
   - Provided direct point-by-point comparison
   - Explained WHY recent information takes precedence"""
        
        elif conflict_type == ConflictType.FACTUAL:
            content = f"""ROUND {round_number} REVISED PROPOSAL:

Addressing critiques: Direct evidence comparison and authority evaluation.

Revised Analysis for "{query}":

1. FACTUAL COMPARISON:
   - Claim A ({source_a}): {claim_a[:80]}...
   - Claim B ({source_b}): {claim_b[:80]}...
   - These make mutually incompatible factual assertions

2. EVIDENCE STRENGTH:
   - Evaluate source authority and expertise
   - Consider recency and relevance to the specific question
   - Assess specificity of information provided

3. REVISED RECOMMENDATION:
   Weigh evidence based on:
   - Source credibility and expertise in the domain
   - Specificity and directness of the information
   - Corroboration with known facts
   
   Final Position: Recommend the claim with stronger evidentiary support
   and more authoritative sourcing.

4. Addressing Critiques:
   - Provided direct evidence comparison
   - Included source authority analysis
   - Strengthened logical reasoning"""
        
        else:
            content = f"""ROUND {round_number} REVISED PROPOSAL:

Addressing critiques from previous round.

Revised Analysis for "{query}":

1. CONFLICT ANALYSIS ({conflict_type.value}):
   - {source_a}: {claim_a[:80]}...
   - {source_b}: {claim_b[:80]}...

2. INCORPORATING FEEDBACK:
   {critique_summary[:200]}

3. REVISED POSITION:
   Based on comprehensive analysis of both sources and addressing
   the identified weaknesses, the revised recommendation considers:
   - Source authority and reliability
   - Temporal relevance
   - Specificity of information
   - Logical consistency

4. FINAL RECOMMENDATION:
   The answer should synthesize the valid elements from both sources
   while resolving the core conflict through careful evidence evaluation."""
        
        return DebateArgument(
            round_number=round_number,
            role="reviser",
            agent_id="reviser",
            content=content,
            evidence_refs=[evidence_a.agent_id, evidence_b.agent_id],
            conflict_type=conflict_type,
            strategy_used=f"{conflict_type.value}_revision",
        )
    
    def _llm_revise(
        self,
        query: str,
        proposal: DebateArgument,
        critiques: List[DebateArgument],
        evidence_a: Evidence,
        evidence_b: Evidence,
        conflict_type: ConflictType,
        round_number: int,
    ) -> DebateArgument:
        """Generate revision using LLM."""
        try:
            claim_a = evidence_a.claims[0].text if evidence_a.claims else "No claims"
            claim_b = evidence_b.claims[0].text if evidence_b.claims else "No claims"
            source_a = evidence_a.sources[0].title if evidence_a.sources else "Unknown"
            source_b = evidence_b.sources[0].title if evidence_b.sources else "Unknown"
            
            critiques_text = "\n\n".join([
                f"Critique {i+1}:\n{c.content}" 
                for i, c in enumerate(critiques)
            ])
            
            prompt = f"""You are a debate reviser. Your job is to produce an improved argument
that addresses the critiques while maintaining the strongest points.

QUESTION: {query}
CONFLICT TYPE: {conflict_type.value}

ORIGINAL CLAIMS:
- {source_a}: {claim_a}
- {source_b}: {claim_b}

ORIGINAL PROPOSAL:
{proposal.content}

CRITIQUES TO ADDRESS:
{critiques_text}

Generate a revised argument (Round {round_number}) that:
1. Directly addresses each critique
2. Incorporates the valid points from critiques
3. Strengthens weak areas
4. Provides a more complete and well-reasoned position
5. References specific evidence

Your revised proposal:"""
            
            response = self.llm_client.chat(prompt)
            
            return DebateArgument(
                round_number=round_number,
                role="reviser",
                agent_id="reviser",
                content=response,
                evidence_refs=[evidence_a.agent_id, evidence_b.agent_id],
                conflict_type=conflict_type,
                strategy_used=f"{conflict_type.value}_revision",
            )
            
        except Exception as e:
            logger.error(f"LLM revision generation failed: {e}")
            return self._demo_revise(
                query, proposal, critiques, evidence_a, evidence_b,
                conflict_type, round_number
            )