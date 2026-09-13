"""
Metrics calculator for evaluating the Multi-Agent Debate system.
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter
import pandas as pd
from loguru import logger

from evidence.models import ConflictType, ExecutionTrace


class MetricsCalculator:
    """
    Calculates evaluation metrics for the MAD system.
    """
    
    def __init__(self):
        self.results: List[ExecutionTrace] = []
    
    def add_result(self, trace: ExecutionTrace) -> None:
        """Add an execution trace to the results."""
        self.results.append(trace)
    
    def add_results(self, traces: List[ExecutionTrace]) -> None:
        """Add multiple execution traces."""
        self.results.extend(traces)
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate all evaluation metrics.
        
        Returns:
            Dictionary of metrics
        """
        if not self.results:
            return {"error": "No results to calculate metrics"}
        
        total = len(self.results)
        # Measured answer accuracy: token recall against the labelled reference.
        answer_correct = 0
        answer_total = 0
        for trace in self.results:
            expected = trace.router_result.get("expected_answer")
            actual = trace.final_answer.answer if trace.final_answer else ""
            if expected:
                answer_total += 1
                expected_tokens = set(re.findall(r"[a-z0-9]+", expected.lower()))
                actual_tokens = set(re.findall(r"[a-z0-9]+", actual.lower()))
                if expected_tokens and len(expected_tokens & actual_tokens) / len(expected_tokens) >= 0.5:
                    answer_correct += 1
        answer_accuracy = answer_correct / answer_total if answer_total else 0.0
        
        # Basic counts
        debate_triggered = sum(1 for r in self.results if r.debate_triggered)
        evidence_agreement = sum(1 for r in self.results if not r.debate_triggered)
        
        # Conflict types
        conflict_types = []
        for r in self.results:
            if r.conflict:
                conflict_types.append(r.conflict.type.value)
        
        # Latency
        latencies = [r.metrics.get("latency_seconds", 0) for r in self.results]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        # LLM calls
        llm_calls = [r.metrics.get("total_llm_calls", 0) for r in self.results]
        avg_llm_calls = sum(llm_calls) / len(llm_calls) if llm_calls else 0
        
        # Debate rounds
        debate_rounds = [r.metrics.get("debate_rounds", 0) for r in self.results if r.debate_triggered]
        avg_debate_rounds = sum(debate_rounds) / len(debate_rounds) if debate_rounds else 0
        
        # Conflict detection accuracy
        correct_conflict_detection = 0
        for r in self.results:
            # Check if detection matches expectation
            expected_conflict = r.router_result.get("expected_conflict", None)
            if expected_conflict is not None:
                detected_conflict = r.debate_triggered
                if detected_conflict == expected_conflict:
                    correct_conflict_detection += 1
        
        conflict_detection_accuracy = (
            correct_conflict_detection / total if total > 0 else 0
        )
        
        # Conflict classification accuracy
        correct_classification = 0
        total_classified = 0
        for r in self.results:
            if r.conflict and r.router_result.get("expected_conflict_type"):
                total_classified += 1
                if r.conflict.type.value == r.router_result["expected_conflict_type"]:
                    correct_classification += 1
        
        conflict_classification_accuracy = (
            correct_classification / total_classified if total_classified > 0 else 0
        )
        
        # Precision, Recall, F1 for conflict detection
        true_positives = sum(
            1 for r in self.results 
            if r.debate_triggered and r.router_result.get("expected_conflict", False)
        )
        false_positives = sum(
            1 for r in self.results 
            if r.debate_triggered and not r.router_result.get("expected_conflict", False)
        )
        false_negatives = sum(
            1 for r in self.results 
            if not r.debate_triggered and r.router_result.get("expected_conflict", False)
        )
        true_negatives = sum(
            1 for r in self.results 
            if not r.debate_triggered and not r.router_result.get("expected_conflict", False)
        )
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            "total_samples": total,
            "answer_accuracy": answer_accuracy,
            "debate_trigger_rate": debate_triggered / total if total > 0 else 0,
            "evidence_agreement_rate": evidence_agreement / total if total > 0 else 0,
            "debate_triggered": debate_triggered,
            "evidence_agreement": evidence_agreement,
            "conflict_detection_accuracy": conflict_detection_accuracy,
            "conflict_classification_accuracy": conflict_classification_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_latency_seconds": avg_latency,
            "avg_llm_calls": avg_llm_calls,
            "avg_debate_rounds": avg_debate_rounds,
            "conflict_type_distribution": dict(Counter(conflict_types)),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
        }
        
        return metrics
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        data = []
        for trace in self.results:
            row = {
                "query": trace.query,
                "debate_triggered": trace.debate_triggered,
                "conflict_type": trace.conflict.type.value if trace.conflict else None,
                "latency": trace.metrics.get("latency_seconds", 0),
                "llm_calls": trace.metrics.get("total_llm_calls", 0),
                "debate_rounds": trace.metrics.get("debate_rounds", 0),
                "contradiction_score": trace.contradiction_score,
                "judge_winner": trace.judge_result.winner if trace.judge_result else None,
                "judge_confidence": trace.judge_result.confidence if trace.judge_result else None,
                "error": trace.error,
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def save_metrics(self, filepath: str) -> None:
        """Save metrics to JSON file."""
        metrics = self.calculate_metrics()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {filepath}")
    
    def save_results_csv(self, filepath: str) -> None:
        """Save results to CSV file."""
        df = self.to_dataframe()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(path, index=False)
        logger.info(f"Results saved to {filepath}")
    
    def print_comparison_table(
        self,
        single_llm_metrics: Dict[str, Any],
        standard_mad_metrics: Dict[str, Any],
        proposed_metrics: Dict[str, Any],
    ) -> str:
        """
        Print a comparison table of different systems.
        
        Args:
            single_llm_metrics: Metrics for single LLM baseline
            standard_mad_metrics: Metrics for standard MAD baseline
            proposed_metrics: Metrics for proposed method
            
        Returns:
            Formatted table string
        """
        lines = []
        lines.append("")
        lines.append("=" * 70)
        lines.append("SYSTEM COMPARISON")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"{'System':<25} {'Accuracy':<12} {'Debate Rate':<15} {'LLM Calls':<12}")
        lines.append("-" * 70)
        
        # Single LLM
        acc1 = single_llm_metrics.get("accuracy", 0) * 100
        lines.append(f"{'Single LLM':<25} {acc1:>6.1f}%     {'0%':<15} {'1':<12}")
        
        # Standard MAD
        acc2 = standard_mad_metrics.get("accuracy", 0) * 100
        llm_calls2 = standard_mad_metrics.get("avg_llm_calls", 0)
        lines.append(f"{'Standard MAD':<25} {acc2:>6.1f}%     {'100%':<15} {llm_calls2:<12.1f}")
        
        # Proposed MAD
        acc3 = proposed_metrics.get("accuracy", 0) * 100
        debate_rate = proposed_metrics.get("debate_trigger_rate", 0) * 100
        llm_calls3 = proposed_metrics.get("avg_llm_calls", 0)
        lines.append(f"{'Proposed MAD':<25} {acc3:>6.1f}%     {debate_rate:>6.1f}%       {llm_calls3:<12.1f}")
        
        lines.append("-" * 70)
        lines.append("")
        
        # Additional metrics
        lines.append("ADDITIONAL METRICS (Proposed MAD):")
        lines.append(f"  Conflict Detection Accuracy: {proposed_metrics.get('conflict_detection_accuracy', 0) * 100:.1f}%")
        lines.append(f"  Conflict Classification Accuracy: {proposed_metrics.get('conflict_classification_accuracy', 0) * 100:.1f}%")
        lines.append(f"  Precision: {proposed_metrics.get('precision', 0) * 100:.1f}%")
        lines.append(f"  Recall: {proposed_metrics.get('recall', 0) * 100:.1f}%")
        lines.append(f"  F1: {proposed_metrics.get('f1', 0) * 100:.1f}%")
        lines.append(f"  Avg Latency: {proposed_metrics.get('avg_latency_seconds', 0):.2f}s")
        lines.append("")
        
        return "\n".join(lines)


def get_metrics_calculator() -> MetricsCalculator:
    """Get or create metrics calculator instance."""
    return MetricsCalculator()