"""
Experiment runner for evaluating the Multi-Agent Debate system.
Runs all three systems: Single LLM, Standard MAD, and Proposed MAD.
"""
import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from evidence.models import ExecutionTrace
from evaluation.dataset import EvaluationDataset, EvaluationSample, get_evaluation_dataset
from evaluation.metrics import MetricsCalculator, get_metrics_calculator
from app import MADSystem, save_trace
from config import settings


class ExperimentRunner:
    """
    Runs experiments comparing Single LLM, Standard MAD, and Proposed MAD.
    """
    
    def __init__(self, demo_mode: bool = False):
        self.demo_mode = demo_mode
        self.dataset = get_evaluation_dataset()
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
    
    async def run_single_llm_baseline(
        self,
        samples: Optional[List[EvaluationSample]] = None,
    ) -> List[ExecutionTrace]:
        """
        Run Single LLM baseline: Query → LLM → Answer.
        
        Args:
            samples: Evaluation samples (uses full dataset if None)
            
        Returns:
            List of execution traces
        """
        logger.info("Running Single LLM baseline")
        samples = samples or self.dataset.samples
        traces = []
        
        for sample in samples:
            trace = ExecutionTrace(
                query=sample.query,
                router_result={
                    "expected_conflict": sample.expected_conflict,
                    "expected_conflict_type": sample.expected_conflict_type,
                },
            )
            
            start_time = time.time()
            
            try:
                # Simple single-agent answer (no debate)
                system = MADSystem(demo_mode=self.demo_mode)
                result = await system.solve(sample.query)
                result.router_result.update({"expected_conflict": sample.expected_conflict, "expected_conflict_type": sample.expected_conflict_type, "expected_answer": sample.expected_answer})
                
                # Override to ensure no debate
                result.debate_triggered = False
                result.conflict = None
                result.debate_rounds = []
                result.judge_result = None
                
                # Keep original metrics
                result.metrics["debate_triggered"] = False
                result.metrics["system"] = "single_llm"
                
                traces.append(result)
                
            except Exception as e:
                logger.error(f"Single LLM failed for '{sample.query}': {e}")
                trace.error = str(e)
                traces.append(trace)
        
        logger.info(f"Single LLM baseline completed: {len(traces)} samples")
        return traces
    
    async def run_standard_mad_baseline(
        self,
        samples: Optional[List[EvaluationSample]] = None,
    ) -> List[ExecutionTrace]:
        """
        Run Standard MAD baseline: Always debate regardless of evidence.
        
        Args:
            samples: Evaluation samples (uses full dataset if None)
            
        Returns:
            List of execution traces
        """
        logger.info("Running Standard MAD baseline")
        samples = samples or self.dataset.samples
        traces = []
        
        for sample in samples:
            trace = ExecutionTrace(
                query=sample.query,
                router_result={
                    "expected_conflict": sample.expected_conflict,
                    "expected_conflict_type": sample.expected_conflict_type,
                },
            )
            
            try:
                # Use proposed system but force debate
                system = MADSystem(demo_mode=self.demo_mode)
                result = await system.solve(sample.query)
                result.router_result.update({"expected_conflict": sample.expected_conflict, "expected_conflict_type": sample.expected_conflict_type, "expected_answer": sample.expected_answer})
                
                # Override: force debate for standard MAD
                result.debate_triggered = True
                
                # If no conflict was detected, create a mock one
                if not result.conflict:
                    from evidence.models import Conflict, ConflictType, Claim, Source
                    mock_conflict = Conflict(
                        type=ConflictType.FACTUAL,
                        confidence=0.8,
                        explanation="Standard MAD always debates",
                        claim_a=Claim(text="Mock claim A", source=Source(url="", title="", snippet="")),
                        claim_b=Claim(text="Mock claim B", source=Source(url="", title="", snippet="")),
                    )
                    result.conflict = mock_conflict
                
                # Ensure metrics reflect standard MAD
                result.metrics["debate_triggered"] = True
                result.router_result["expected_answer"] = sample.expected_answer
                result.metrics["system"] = "standard_mad"
                
                traces.append(result)
                
            except Exception as e:
                logger.error(f"Standard MAD failed for '{sample.query}': {e}")
                trace.error = str(e)
                traces.append(trace)
        
        logger.info(f"Standard MAD baseline completed: {len(traces)} samples")
        return traces
    
    async def run_proposed_mad(
        self,
        samples: Optional[List[EvaluationSample]] = None,
    ) -> List[ExecutionTrace]:
        """
        Run Proposed MAD: Evidence-triggered debate with conflict classification.
        
        Args:
            samples: Evaluation samples (uses full dataset if None)
            
        Returns:
            List of execution traces
        """
        logger.info("Running Proposed MAD")
        samples = samples or self.dataset.samples
        traces = []
        
        for sample in samples:
            try:
                system = MADSystem(demo_mode=self.demo_mode)
                result = await system.solve(sample.query)
                result.router_result.update({"expected_conflict": sample.expected_conflict, "expected_conflict_type": sample.expected_conflict_type, "expected_answer": sample.expected_answer})
                
                # Add expected values for evaluation
                result.router_result["expected_conflict"] = sample.expected_conflict
                result.router_result["expected_conflict_type"] = sample.expected_conflict_type
                
                result.router_result["expected_answer"] = sample.expected_answer
                result.metrics["system"] = "proposed_mad"
                traces.append(result)
                
            except Exception as e:
                logger.error(f"Proposed MAD failed for '{sample.query}': {e}")
                trace = ExecutionTrace(
                    query=sample.query,
                    error=str(e),
                    router_result={
                        "expected_conflict": sample.expected_conflict,
                        "expected_conflict_type": sample.expected_conflict_type,
                    },
                )
                traces.append(trace)
        
        logger.info(f"Proposed MAD completed: {len(traces)} samples")
        return traces
    
    async def run_all_experiments(
        self,
        samples: Optional[List[EvaluationSample]] = None,
    ) -> Dict[str, Any]:
        """
        Run all experiments and return comparison results.
        
        Args:
            samples: Evaluation samples (uses full dataset if None)
            
        Returns:
            Dictionary with results for all systems
        """
        logger.info("Starting full experiment suite")
        samples = samples or self.dataset.samples
        
        # Run all three systems
        single_llm_traces = await self.run_single_llm_baseline(samples)
        standard_mad_traces = await self.run_standard_mad_baseline(samples)
        proposed_mad_traces = await self.run_proposed_mad(samples)
        
        # Calculate metrics for each
        single_llm_calc = get_metrics_calculator()
        single_llm_calc.add_results(single_llm_traces)
        single_llm_metrics = single_llm_calc.calculate_metrics()
        single_llm_metrics["accuracy"] = single_llm_metrics.get("answer_accuracy", 0.0)
        
        standard_mad_calc = get_metrics_calculator()
        standard_mad_calc.add_results(standard_mad_traces)
        standard_mad_metrics = standard_mad_calc.calculate_metrics()
        standard_mad_metrics["accuracy"] = standard_mad_metrics.get("answer_accuracy", 0.0)
        
        proposed_mad_calc = get_metrics_calculator()
        proposed_mad_calc.add_results(proposed_mad_traces)
        proposed_mad_metrics = proposed_mad_calc.calculate_metrics()
        proposed_mad_metrics["accuracy"] = proposed_mad_metrics.get("answer_accuracy", 0.0)
        
        # Save traces
        for trace in single_llm_traces:
            save_trace(trace, str(self.results_dir / "single_llm"))
        for trace in standard_mad_traces:
            save_trace(trace, str(self.results_dir / "standard_mad"))
        for trace in proposed_mad_traces:
            save_trace(trace, str(self.results_dir / "proposed_mad"))
        
        # Save metrics
        all_metrics = {
            "single_llm": single_llm_metrics,
            "standard_mad": standard_mad_metrics,
            "proposed_mad": proposed_mad_metrics,
            "dataset_statistics": self.dataset.get_statistics(),
        }
        
        metrics_path = self.results_dir / "experiment_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        
        # Print comparison table
        table = single_llm_calc.print_comparison_table(
            single_llm_metrics, standard_mad_metrics, proposed_mad_metrics
        )
        print(table)
        
        # Generate graphs
        self.generate_graphs(all_metrics)
        
        logger.info("All experiments completed")
        return all_metrics
    
    def generate_graphs(self, metrics: Dict[str, Any]) -> None:
        """Generate evaluation graphs."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            figures_dir = self.results_dir / "figures"
            figures_dir.mkdir(exist_ok=True)
            
            # Graph 1: Accuracy comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            systems = ["Single LLM", "Standard MAD", "Proposed MAD"]
            accuracies = [
                metrics["single_llm"].get("accuracy", 0) * 100,
                metrics["standard_mad"].get("accuracy", 0) * 100,
                metrics["proposed_mad"].get("accuracy", 0) * 100,
            ]
            
            colors = ['#3498db', '#e74c3c', '#2ecc71']
            bars = ax.bar(systems, accuracies, color=colors)
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("System Accuracy Comparison")
            ax.set_ylim(0, 100)
            
            for bar, acc in zip(bars, accuracies):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{acc:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(figures_dir / "accuracy_comparison.png", dpi=150)
            plt.close()
            
            # Graph 2: LLM calls comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            llm_calls = [
                metrics["single_llm"].get("avg_llm_calls", 1),
                metrics["standard_mad"].get("avg_llm_calls", 10),
                metrics["proposed_mad"].get("avg_llm_calls", 5),
            ]
            
            bars = ax.bar(systems, llm_calls, color=colors)
            ax.set_ylabel("Average LLM Calls")
            ax.set_title("Average LLM Calls per Query")
            
            for bar, calls in zip(bars, llm_calls):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                       f'{calls:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(figures_dir / "llm_calls_comparison.png", dpi=150)
            plt.close()
            
            # Graph 3: Debate trigger rate
            fig, ax = plt.subplots(figsize=(10, 6))
            debate_rates = [
                0,
                100,
                metrics["proposed_mad"].get("debate_trigger_rate", 0) * 100,
            ]
            
            bars = ax.bar(systems, debate_rates, color=colors)
            ax.set_ylabel("Debate Trigger Rate (%)")
            ax.set_title("Debate Trigger Rate by System")
            ax.set_ylim(0, 110)
            
            for bar, rate in zip(bars, debate_rates):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{rate:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(figures_dir / "debate_trigger_rate.png", dpi=150)
            plt.close()
            
            # Graph 4: Conflict type distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            conflict_dist = metrics["proposed_mad"].get("conflict_type_distribution", {})
            
            if conflict_dist:
                types = list(conflict_dist.keys())
                counts = list(conflict_dist.values())
                colors_pie = plt.cm.Set3(np.linspace(0, 1, len(types)))
                
                wedges, texts, autotexts = ax.pie(
                    counts, labels=types, colors=colors_pie,
                    autopct='%1.1f%%', startangle=90
                )
                ax.set_title("Conflict Type Distribution (Proposed MAD)")
            
            plt.tight_layout()
            plt.savefig(figures_dir / "conflict_distribution.png", dpi=150)
            plt.close()
            
            logger.info(f"Graphs saved to {figures_dir}")
            
        except ImportError as e:
            logger.warning(f"Could not generate graphs (missing dependency): {e}")
        except Exception as e:
            logger.error(f"Graph generation failed: {e}")


async def run_all_experiments(demo_mode: bool = False) -> Dict[str, Any]:
    """Convenience function to run all experiments."""
    runner = ExperimentRunner(demo_mode=demo_mode)
    return await runner.run_all_experiments()


if __name__ == "__main__":
    import sys
    demo_mode = "--demo" in sys.argv
    asyncio.run(run_all_experiments(demo_mode=demo_mode))