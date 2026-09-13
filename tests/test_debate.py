"""
Tests for debate components and the evidence gate.
"""
import pytest
import asyncio
from evidence.models import (
    Evidence, Claim, Source, Conflict, ConflictType, DebateRound
)
from debate.proposer import Proposer
from debate.critic import Critic
from debate.reviser import Reviser
from debate.manager import DebateManager
from debate.judge import Judge
from app import MADSystem


class TestProposer:
    """Test the Proposer agent."""
    
    def test_proposer_creates_argument(self):
        """Test that proposer creates an argument."""
        proposer = Proposer()
        
        source_a = Source(url="https://example.com/a", title="Source A", snippet="Snippet")
        source_b = Source(url="https://example.com/b", title="Source B", snippet="Snippet")
        
        evidence_a = Evidence(
            agent_id="agent_a",
            query="Test query",
            answer="Answer A",
            claims=[Claim(text="Claim A text", source=source_a)],
            sources=[source_a],
        )
        
        evidence_b = Evidence(
            agent_id="agent_b",
            query="Test query",
            answer="Answer B",
            claims=[Claim(text="Claim B text", source=source_b)],
            sources=[source_b],
        )
        
        conflict = Conflict(
            type=ConflictType.TEMPORAL,
            confidence=0.85,
            explanation="Temporal conflict",
            claim_a=evidence_a.claims[0],
            claim_b=evidence_b.claims[0],
        )
        
        argument = proposer.propose(
            query="Test query",
            evidence_a=evidence_a,
            evidence_b=evidence_b,
            conflict_type=ConflictType.TEMPORAL,
            conflict_explanation="Temporal conflict",
            round_number=1,
        )
        
        assert argument is not None
        assert argument.role == "proposer"
        assert argument.round_number == 1
        assert len(argument.content) > 0


class TestCritic:
    """Test the Critic agent."""
    
    def test_critic_creates_critiques(self):
        """Test that critic creates critiques."""
        critic = Critic()
        
        source_a = Source(url="https://example.com/a", title="Source A", snippet="Snippet")
        source_b = Source(url="https://example.com/b", title="Source B", snippet="Snippet")
        
        evidence_a = Evidence(
            agent_id="agent_a",
            query="Test query",
            answer="Answer A",
            claims=[Claim(text="Claim A text", source=source_a)],
            sources=[source_a],
        )
        
        evidence_b = Evidence(
            agent_id="agent_b",
            query="Test query",
            answer="Answer B",
            claims=[Claim(text="Claim B text", source=source_b)],
            sources=[source_b],
        )
        
        # Create a proposal to critique
        proposer = Proposer()
        proposal = proposer.propose(
            query="Test query",
            evidence_a=evidence_a,
            evidence_b=evidence_b,
            conflict_type=ConflictType.FACTUAL,
            conflict_explanation="Factual conflict",
            round_number=1,
        )
        
        critiques = critic.critique(
            query="Test query",
            proposal=proposal,
            evidence_a=evidence_a,
            evidence_b=evidence_b,
            conflict_type=ConflictType.FACTUAL,
            round_number=1,
        )
        
        assert isinstance(critiques, list)
        assert len(critiques) > 0
        assert all(c.role == "critic" for c in critiques)


class TestReviser:
    """Test the Reviser agent."""
    
    def test_reviser_creates_revision(self):
        """Test that reviser creates a revision."""
        reviser = Reviser()
        
        source_a = Source(url="https://example.com/a", title="Source A", snippet="Snippet")
        source_b = Source(url="https://example.com/b", title="Source B", snippet="Snippet")
        
        evidence_a = Evidence(
            agent_id="agent_a",
            query="Test query",
            answer="Answer A",
            claims=[Claim(text="Claim A text", source=source_a)],
            sources=[source_a],
        )
        
        evidence_b = Evidence(
            agent_id="agent_b",
            query="Test query",
            answer="Answer B",
            claims=[Claim(text="Claim B text", source=source_b)],
            sources=[source_b],
        )
        
        # Create proposal and critiques
        proposer = Proposer()
        proposal = proposer.propose(
            query="Test query",
            evidence_a=evidence_a,
            evidence_b=evidence_b,
            conflict_type=ConflictType.VERSION,
            conflict_explanation="Version conflict",
            round_number=1,
        )
        
        critic = Critic()
        critiques = critic.critique(
            query="Test query",
            proposal=proposal,
            evidence_a=evidence_a,
            evidence_b=evidence_b,
            conflict_type=ConflictType.VERSION,
            round_number=1,
        )
        
        # Revise
        revision = reviser.revise(
            query="Test query",
            proposal=proposal,
            critiques=critiques,
            evidence_a=evidence_a,
            evidence_b=evidence_b,
            conflict_type=ConflictType.VERSION,
            round_number=1,
        )
        
        assert revision is not None
        assert revision.role == "reviser"
        assert len(revision.content) > 0


class TestDebateManager:
    """Test the DebateManager."""
    
    def test_manager_runs_debate(self):
        """Test that manager runs debate correctly."""
        manager = DebateManager(max_rounds=2)
        
        source_a = Source(url="https://example.com/a", title="Source A", snippet="Snippet")
        source_b = Source(url="https://example.com/b", title="Source B", snippet="Snippet")
        
        evidence_a = Evidence(
            agent_id="agent_a",
            query="Test query",
            answer="Answer A",
            claims=[Claim(text="Claim A text", source=source_a)],
            sources=[source_a],
        )
        
        evidence_b = Evidence(
            agent_id="agent_b",
            query="Test query",
            answer="Answer B",
            claims=[Claim(text="Claim B text", source=source_b)],
            sources=[source_b],
        )
        
        conflict = Conflict(
            type=ConflictType.FACTUAL,
            confidence=0.85,
            explanation="Factual conflict",
            claim_a=evidence_a.claims[0],
            claim_b=evidence_b.claims[0],
        )
        
        rounds = manager.run_debate(
            query="Test query",
            evidence_a=evidence_a,
            evidence_b=evidence_b,
            conflict=conflict,
        )
        
        assert isinstance(rounds, list)
        assert len(rounds) == 2
        assert all(isinstance(r, DebateRound) for r in rounds)
        
        # Each round should have proposer, critiques, and reviser
        for r in rounds:
            assert r.proposer_argument is not None
            assert len(r.critic_arguments) > 0
            assert r.reviser_argument is not None


class TestJudge:
    """Test the Judge."""
    
    def test_judge_evaluates(self):
        """Test that judge evaluates debate."""
        judge = Judge()
        
        source_a = Source(url="https://example.com/a", title="Source A", snippet="Snippet")
        source_b = Source(url="https://example.com/b", title="Source B", snippet="Snippet")
        
        evidence_a = Evidence(
            agent_id="agent_a",
            query="Test query",
            answer="Answer A",
            claims=[Claim(text="Claim A text", source=source_a)],
            sources=[source_a],
        )
        
        evidence_b = Evidence(
            agent_id="agent_b",
            query="Test query",
            answer="Answer B",
            claims=[Claim(text="Claim B text", source=source_b)],
            sources=[source_b],
        )
        
        conflict = Conflict(
            type=ConflictType.TEMPORAL,
            confidence=0.85,
            explanation="Temporal conflict",
            claim_a=evidence_a.claims[0],
            claim_b=evidence_b.claims[0],
        )
        
        # Run debate
        manager = DebateManager(max_rounds=1)
        rounds = manager.run_debate(
            query="Test query",
            evidence_a=evidence_a,
            evidence_b=evidence_b,
            conflict=conflict,
        )
        
        # Evaluate
        result = judge.evaluate(
            query="Test query",
            evidence_a=evidence_a,
            evidence_b=evidence_b,
            debate_rounds=rounds,
            conflict_type=ConflictType.TEMPORAL,
        )
        
        assert result is not None
        assert result.winner in ["candidate_1", "candidate_2"]
        assert 0 <= result.evidence_score <= 1
        assert 0 <= result.confidence <= 1


class TestEvidenceGate:
    """Test the evidence gate logic (critical tests)."""
    
    def test_agreement_skips_debate(self):
        """Test that evidence agreement skips debate."""
        # Create agreeing evidence
        source = Source(url="https://example.com", title="Source", snippet="Snippet")
        
        evidence_a = Evidence(
            agent_id="agent_a",
            query="What is Python?",
            answer="Python is a programming language",
            claims=[Claim(text="Python is a programming language", source=source)],
            sources=[source],
        )
        
        evidence_b = Evidence(
            agent_id="agent_b",
            query="What is Python?",
            answer="Python is a programming language",
            claims=[Claim(text="Python is a programming language", source=source)],
            sources=[source],
        )
        
        # Run the system
        system = MADSystem(demo_mode=True)
        trace = asyncio.run(system.solve("What is Python?"))
        
        # Should NOT trigger debate
        assert trace.debate_triggered is False
        assert trace.conflict is None
        assert len(trace.debate_rounds) == 0
    
    def test_disagreement_triggers_debate(self):
        """Test that evidence disagreement triggers debate."""
        system = MADSystem(demo_mode=True)
        
        # The demo system should detect temporal conflict
        trace = asyncio.run(system.solve("When was the policy introduced?"))
        
        # Should trigger debate
        assert trace.debate_triggered is True
        assert trace.conflict is not None
        assert len(trace.debate_rounds) > 0
    
    def test_debate_triggered_logged(self):
        """Test that debate_triggered is logged in metrics."""
        system = MADSystem(demo_mode=True)
        trace = asyncio.run(system.solve("Test query"))
        
        assert "debate_triggered" in trace.metrics
        assert isinstance(trace.metrics["debate_triggered"], bool)