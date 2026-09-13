"""
Main application for the Multi-Agent Debate framework.
Contains the solve() pipeline and CLI interface.
"""
import asyncio
import json
import time
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from loguru import logger

from config import settings
from agents import router
from evidence import (
    Evidence, EvidenceComparison, Conflict, ConflictType,
    ExecutionTrace, FinalAnswer
)
from evidence.comparator import EvidenceComparator
from evidence.disagreement_detector import get_nli_detector, MockNLIDetector
from conflict.classifier import ConflictClassifier, get_conflict_classifier
from debate.manager import DebateManager, get_debate_manager
from debate.judge import Judge, get_judge
from output.generator import AnswerGenerator, get_answer_generator
from llm_client import LLMClient, get_llm_client


class MADSystem:
    """
    Multi-Agent Debate system with evidence-triggered debate
    and conflict-type-aware debate strategy.
    """
    
    def __init__(self, demo_mode: bool = False):
        self.demo_mode = demo_mode or settings.demo_mode
        self.llm_client = get_llm_client(demo_mode=self.demo_mode)
        self.nli_detector = (MockNLIDetector(threshold=settings.disagreement_threshold) if (self.demo_mode or settings.nli_mode == "mock") else get_nli_detector())
        self.conflict_classifier = get_conflict_classifier(self.llm_client)
        if self.demo_mode:
            self.conflict_classifier.demo_mode = True
        self.debate_manager = get_debate_manager(self.llm_client)
        if self.demo_mode:
            self.debate_manager.demo_mode = True
            self.debate_manager.proposer.demo_mode = True
            self.debate_manager.critic.demo_mode = True
            self.debate_manager.reviser.demo_mode = True
        self.judge = get_judge(self.llm_client)
        if self.demo_mode:
            self.judge.demo_mode = True
        self.answer_generator = get_answer_generator(self.llm_client)
        self.evidence_comparator = EvidenceComparator(self.nli_detector)
        
        # Initialize router
        router.initialize(self.demo_mode)
    
    async def solve(self, query: str) -> ExecutionTrace:
        """
        Main solve pipeline.
        
        Args:
            query: User question to answer
            
        Returns:
            ExecutionTrace with complete results
        """
        trace = ExecutionTrace(query=query)
        start_time = time.time()
        
        try:
            logger.info(f"Solving: {query}")
            
            # 1. Route query to agents
            routing_result = await router.execute(query)
            trace.router_result = routing_result
            
            # Get evidence from agents
            evidence_list = []
            for result in routing_result.get("evidence", []):
                if "evidence" in result:
                    ev = Evidence(**result["evidence"])
                    evidence_list.append(ev)
            
            if len(evidence_list) < 2:
                logger.warning("Less than 2 agents returned evidence, using defaults")
                evidence_list = self._create_default_evidence(query)
            
            trace.agents = evidence_list
            
            evidence_a = evidence_list[0]
            evidence_b = evidence_list[1]
            
            # 2. Compare evidence
            comparison = self.evidence_comparator.compare(evidence_a, evidence_b)
            trace.evidence_comparison = comparison
            trace.disagreement_detected = comparison.has_disagreement
            trace.contradiction_score = comparison.contradiction_score
            
            # 3. Evidence Gate: Check if debate is needed
            if not comparison.has_disagreement:
                # AGREE path: Skip debate, produce direct answer
                logger.info("Evidence AGREE - skipping debate")
                trace.debate_triggered = False
                
                final_answer = self.answer_generator.generate(
                    query=query,
                    evidence_a=evidence_a,
                    evidence_b=evidence_b,
                    debate_triggered=False,
                    conflict=None,
                    debate_rounds=[],
                    judge_result=None,
                )
                trace.final_answer = final_answer
                
            else:
                # DISAGREE path: Classify conflict and run debate
                logger.info("Evidence DISAGREE - triggering debate")
                trace.debate_triggered = True
                
                # 4. Classify conflict type
                conflict_result = self.conflict_classifier.classify(
                    claim_a=comparison.conflicts[0]["claim_a"] if comparison.conflicts else "",
                    claim_b=comparison.conflicts[0]["claim_b"] if comparison.conflicts else "",
                    query=query,
                )
                
                conflict = Conflict(
                    type=ConflictType(conflict_result["type"]),
                    confidence=conflict_result["confidence"],
                    explanation=conflict_result["explanation"],
                    claim_a=evidence_a.claims[0] if evidence_a.claims else None,
                    claim_b=evidence_b.claims[0] if evidence_b.claims else None,
                )
                trace.conflict = conflict
                
                # 5. Run adaptive debate
                debate_rounds = self.debate_manager.run_debate(
                    query=query,
                    evidence_a=evidence_a,
                    evidence_b=evidence_b,
                    conflict=conflict,
                )
                trace.debate_rounds = debate_rounds
                
                # 6. Judge evaluates debate
                judge_result = self.judge.evaluate(
                    query=query,
                    evidence_a=evidence_a,
                    evidence_b=evidence_b,
                    debate_rounds=debate_rounds,
                    conflict_type=conflict.type,
                )
                trace.judge_result = judge_result
                
                # 7. Generate final answer
                final_answer = self.answer_generator.generate(
                    query=query,
                    evidence_a=evidence_a,
                    evidence_b=evidence_b,
                    debate_triggered=True,
                    conflict=conflict,
                    debate_rounds=debate_rounds,
                    judge_result=judge_result,
                )
                trace.final_answer = final_answer
            
            # Calculate metrics
            end_time = time.time()
            trace.metrics = {
                "latency_seconds": end_time - start_time,
                "total_llm_calls": self._count_llm_calls(),
                "evidence_agreement": not comparison.has_disagreement,
                "contradiction_score": comparison.contradiction_score,
                "debate_triggered": trace.debate_triggered,
                "conflict_type": trace.conflict.type.value if trace.conflict else None,
                "debate_rounds": len(trace.debate_rounds),
            }
            trace.completed_at = datetime.now()
            save_trace(trace)
            
            logger.info(
                f"Solved in {trace.metrics['latency_seconds']:.2f}s, "
                f"debate_triggered={trace.debate_triggered}, "
                f"conflict_type={trace.conflict.type.value if trace.conflict else 'none'}"
            )
            
            return trace
            
        except Exception as e:
            logger.error(f"Solve pipeline failed: {e}")
            trace.error = str(e)
            trace.completed_at = datetime.now()
            save_trace(trace)
            return trace
    
    def _count_llm_calls(self) -> int:
        """Count total LLM calls across all components."""
        total = 0
        if self.llm_client:
            total += self.llm_client.call_count
        if self.debate_manager:
            total += 0
        if self.judge:
            total += 0
        return total
    
    def _create_default_evidence(self, query: str) -> list:
        """Create default evidence when agents fail."""
        from evidence.models import Source, Claim
        
        default_source = Source(
            url="https://example.com/default",
            title="Default Source",
            snippet=f"Default evidence for query: {query}",
        )
        
        evidence = Evidence(
            agent_id="default_agent_a",
            query=query,
            answer=f"Default answer for: {query}",
            claims=[
                Claim(
                    text=f"Default claim for {query}",
                    source=default_source,
                )
            ],
            sources=[default_source],
            retrieved_text=f"Default evidence for {query}",
        )
        
        return [evidence, evidence]


def save_trace(trace: ExecutionTrace, results_dir: str = "results") -> str:
    """Save execution trace to JSON file."""
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)
    
    filename = f"trace_{trace.trace_id[:8]}.json"
    filepath = results_path / filename
    
    with open(filepath, "w") as f:
        json.dump(trace.model_dump(mode="json"), f, indent=2, default=str)
    
    logger.info(f"Trace saved to {filepath}")
    return str(filepath)


async def solve(query: str, demo_mode: bool = False) -> ExecutionTrace:
    """
    Main solve function.
    
    Args:
        query: User question
        demo_mode: Whether to use demo mode
        
    Returns:
        ExecutionTrace with results
    """
    system = MADSystem(demo_mode=demo_mode)
    return await system.solve(query)


def run_interactive():
    """Run interactive CLI mode."""
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    
    console = Console()
    
    console.print(Panel.fit(
        "[bold blue]Multi-Agent Debate System[/]\n"
        "[dim]Evidence-Triggered Debate with Conflict Classification[/]",
        border_style="blue"
    ))
    
    demo_mode = "--demo" in __import__('sys').argv
    
    if demo_mode:
        console.print("[yellow]Running in DEMO MODE[/]", style="yellow")
    
    system = MADSystem(demo_mode=demo_mode)
    
    while True:
        console.print("\n[bold]Enter your query (or 'quit' to exit):[/]")
        query = input("> ").strip()
        
        if query.lower() in ["quit", "exit", "q"]:
            console.print("[dim]Goodbye![/]")
            break
        
        if not query:
            continue
        
        with console.status("[bold green]Processing..."):
            trace = asyncio.run(system.solve(query))
        
        if trace.error:
            console.print(f"[red]Error: {trace.error}[/]")
            continue
        
        # Display results
        console.print("\n" + "=" * 60)
        
        if trace.debate_triggered:
            console.print(f"[bold yellow]Debate Triggered: YES[/]")
            if trace.conflict:
                console.print(f"[yellow]Conflict Type: {trace.conflict.type.value}[/]")
                console.print(f"[yellow]Confidence: {trace.conflict.confidence:.0%}[/]")
        else:
            console.print(f"[bold green]Debate Triggered: NO (evidence agreed)[/]")
        
        console.print(f"[dim]Contradiction Score: {trace.contradiction_score:.3f}[/]")
        console.print(f"[dim]Latency: {trace.metrics.get('latency_seconds', 0):.2f}s[/]")
        
        console.print("\n" + "-" * 60)
        
        if trace.final_answer:
            console.print(Panel(
                trace.final_answer.answer,
                title="Answer",
                border_style="green"
            ))
            
            if trace.final_answer.reasoning:
                console.print(Panel(
                    trace.final_answer.reasoning,
                    title="Reasoning",
                    border_style="dim"
                ))
            
            if trace.final_answer.sources:
                console.print("\n[bold]Sources:[/]")
                for i, source in enumerate(trace.final_answer.sources, 1):
                    console.print(f"  {i}. {source.title}")
                    console.print(f"     [link={source.url}]{source.url}[/link]")


def main():
    """Main entry point."""
    import sys
    
    if "--api" in sys.argv:
        import uvicorn
        from api import app
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        run_interactive()


if __name__ == "__main__":
    main()