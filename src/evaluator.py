"""
Evaluator Module - Stopping conditions and quality checks.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from src.memory import WorkingMemory


@dataclass
class EvaluationResult:
    """Result of an evaluation check."""
    should_continue: bool
    confidence: float
    progress_percentage: float
    reasoning: str
    issues: List[str]


class Evaluator:
    """Evaluates agent progress and determines stopping conditions."""
    
    def __init__(
        self,
        max_iterations: int = 10,
        min_improvement_threshold: float = 0.1,
        max_consecutive_no_progress: int = 2,
        min_confidence_to_stop: float = 0.8
    ):
        self.max_iterations = max_iterations
        self.min_improvement_threshold = min_improvement_threshold
        self.max_consecutive_no_progress = max_consecutive_no_progress
        self.min_confidence_to_stop = min_confidence_to_stop
        self._previous_progress: float = 0.0
        self._no_progress_count: int = 0
    
    def evaluate(
        self,
        memory: WorkingMemory,
        current_iteration: int,
        success_criteria: List[str],
        llm_client: Optional[Any] = None
    ) -> EvaluationResult:
        issues = []
        
        # Check maximum iterations
        if current_iteration >= self.max_iterations:
            return EvaluationResult(
                should_continue=False,
                confidence=1.0,
                progress_percentage=self._estimate_progress(memory, success_criteria),
                reasoning=f"Maximum iterations ({self.max_iterations}) reached",
                issues=["Reached iteration limit"]
            )
        
        progress = self._estimate_progress(memory, success_criteria)
        
        # Check diminishing returns
        improvement = progress - self._previous_progress
        if improvement < self.min_improvement_threshold:
            self._no_progress_count += 1
            issues.append(f"Low improvement: {improvement:.2%}")
        else:
            self._no_progress_count = 0
        
        self._previous_progress = progress
        
        if self._no_progress_count >= self.max_consecutive_no_progress:
            return EvaluationResult(
                should_continue=False,
                confidence=0.7,
                progress_percentage=progress,
                reasoning=f"Diminishing returns detected ({self._no_progress_count} iterations with low improvement)",
                issues=issues
            )
        
        # Check repetition
        if self._detect_repetition(memory):
            issues.append("Repetitive tool usage detected")
            return EvaluationResult(
                should_continue=False,
                confidence=0.6,
                progress_percentage=progress,
                reasoning="Agent appears to be repeating actions",
                issues=issues
            )
        
        # Check goal satisfaction
        if progress >= self.min_confidence_to_stop:
            return EvaluationResult(
                should_continue=False,
                confidence=progress,
                progress_percentage=progress,
                reasoning="Goal appears to be satisfied",
                issues=[]
            )
        
        return EvaluationResult(
            should_continue=True,
            confidence=1.0 - progress,
            progress_percentage=progress,
            reasoning=f"Progress at {progress:.0%}, continuing analysis",
            issues=issues
        )
    
    def _estimate_progress(self, memory: WorkingMemory, success_criteria: List[str]) -> float:
        score = 0.0
        insight_count = len(memory.insights)
        insight_score = min(insight_count / 5, 1.0) * 0.4
        score += insight_score
        
        unique_tools = set(r.tool_name for r in memory.tool_results)
        tool_score = min(len(unique_tools) / 4, 1.0) * 0.3
        score += tool_score
        
        if memory.tool_results:
            success_rate = sum(1 for r in memory.tool_results if r.success) / len(memory.tool_results)
            score += success_rate * 0.3
        
        return min(score, 1.0)
    
    def _detect_repetition(self, memory: WorkingMemory) -> bool:
        recent_tools = [r.tool_name for r in memory.tool_results[-4:]]
        if len(recent_tools) >= 4:
            if recent_tools[-1] == recent_tools[-3] and recent_tools[-2] == recent_tools[-4]:
                return True
            if len(set(recent_tools[-3:])) == 1:
                return True
        return False
    
    def validate_output(self, final_output: Dict[str, Any], goal: str) -> Dict[str, Any]:
        issues = []
        required_fields = ["executive_summary", "key_findings", "recommendations"]
        for field in required_fields:
            if field not in final_output or not final_output[field]:
                issues.append(f"Missing or empty required field: {field}")
        
        if "key_findings" in final_output:
            findings = final_output["key_findings"]
            if isinstance(findings, list) and len(findings) < 2:
                issues.append("Insufficient key findings (fewer than 2)")
        
        return {"valid": len(issues) == 0, "issues": issues, "quality_score": 1.0 - (len(issues) * 0.2)}
    
    def reset(self):
        self._previous_progress = 0.0
        self._no_progress_count = 0
