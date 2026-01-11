"""
Working Memory Module - Transparent, inspectable memory for the agentic system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
import json


class StepType(Enum):
    PLANNING = "planning"
    TOOL_SELECTION = "tool_selection"
    TOOL_EXECUTION = "tool_execution"
    OBSERVATION = "observation"
    EVALUATION = "evaluation"
    SYNTHESIS = "synthesis"


@dataclass
class MemoryEntry:
    """A single entry in working memory."""
    step_number: int
    step_type: StepType
    timestamp: datetime
    content: Dict[str, Any]
    reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "step_type": self.step_type.value,
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "reasoning": self.reasoning,
        }


@dataclass
class Plan:
    """Represents the agent's analysis plan."""
    goal: str
    steps: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "steps": self.steps,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
        }


@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool_name: str
    input_args: Dict[str, Any]
    output: Any
    success: bool
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "input_args": self.input_args,
            "output": self.output if isinstance(self.output, (str, dict, list)) else str(self.output),
            "success": self.success,
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms,
        }


class WorkingMemory:
    """Working memory for the agentic system."""
    
    def __init__(self):
        self._plan: Optional[Plan] = None
        self._entries: List[MemoryEntry] = []
        self._tool_results: List[ToolResult] = []
        self._insights: List[str] = []
        self._step_counter: int = 0
        self._created_at: datetime = datetime.now()
    
    @property
    def plan(self) -> Optional[Plan]:
        return self._plan
    
    @property
    def entries(self) -> List[MemoryEntry]:
        return self._entries.copy()
    
    @property
    def tool_results(self) -> List[ToolResult]:
        return self._tool_results.copy()
    
    @property
    def insights(self) -> List[str]:
        return self._insights.copy()
    
    @property
    def step_count(self) -> int:
        return self._step_counter
    
    def set_plan(self, goal: str, steps: List[str]) -> Plan:
        self._plan = Plan(goal=goal, steps=steps)
        self._add_entry(
            step_type=StepType.PLANNING,
            content={"plan": self._plan.to_dict()},
            reasoning=f"Generated plan with {len(steps)} steps to achieve goal: {goal}"
        )
        return self._plan
    
    def record_tool_selection(self, tool_name: str, reasoning: str, alternatives_considered: Optional[List[str]] = None) -> None:
        self._add_entry(
            step_type=StepType.TOOL_SELECTION,
            content={"selected_tool": tool_name, "alternatives_considered": alternatives_considered or []},
            reasoning=reasoning
        )
    
    def record_tool_result(self, result: ToolResult) -> None:
        self._tool_results.append(result)
        self._add_entry(
            step_type=StepType.TOOL_EXECUTION,
            content=result.to_dict(),
            reasoning=f"Executed {result.tool_name}: {'success' if result.success else 'failed'}"
        )
    
    def record_observation(self, observation: str, data: Optional[Dict] = None) -> None:
        self._add_entry(
            step_type=StepType.OBSERVATION,
            content={"observation": observation, "data": data or {}},
            reasoning=observation
        )
    
    def record_evaluation(self, progress: float, should_continue: bool, reasoning: str, details: Optional[Dict] = None) -> None:
        self._add_entry(
            step_type=StepType.EVALUATION,
            content={"progress": progress, "should_continue": should_continue, "details": details or {}},
            reasoning=reasoning
        )
    
    def add_insight(self, insight: str) -> None:
        self._insights.append(insight)
    
    def get_context_for_llm(self, max_entries: int = 10) -> str:
        context_parts = []
        if self._plan:
            context_parts.append(f"CURRENT PLAN:\n{json.dumps(self._plan.to_dict(), indent=2)}")
        recent_entries = self._entries[-max_entries:]
        if recent_entries:
            entries_text = "\n".join([f"[Step {e.step_number}] {e.step_type.value}: {e.reasoning}" for e in recent_entries])
            context_parts.append(f"RECENT STEPS:\n{entries_text}")
        if self._insights:
            insights_text = "\n".join([f"- {i}" for i in self._insights])
            context_parts.append(f"INSIGHTS DISCOVERED:\n{insights_text}")
        return "\n\n".join(context_parts)
    
    def get_last_tool_result(self) -> Optional[ToolResult]:
        return self._tool_results[-1] if self._tool_results else None
    
    def get_tool_results_by_name(self, tool_name: str) -> List[ToolResult]:
        return [r for r in self._tool_results if r.tool_name == tool_name]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_at": self._created_at.isoformat(),
            "step_count": self._step_counter,
            "plan": self._plan.to_dict() if self._plan else None,
            "entries": [e.to_dict() for e in self._entries],
            "tool_results": [r.to_dict() for r in self._tool_results],
            "insights": self._insights,
        }
    
    def get_reasoning_trace(self) -> List[str]:
        return [entry.reasoning for entry in self._entries]
    
    def clear(self) -> None:
        self._plan = None
        self._entries = []
        self._tool_results = []
        self._insights = []
        self._step_counter = 0
    
    def _add_entry(self, step_type: StepType, content: Dict[str, Any], reasoning: str) -> MemoryEntry:
        self._step_counter += 1
        entry = MemoryEntry(
            step_number=self._step_counter,
            step_type=step_type,
            timestamp=datetime.now(),
            content=content,
            reasoning=reasoning
        )
        self._entries.append(entry)
        return entry
