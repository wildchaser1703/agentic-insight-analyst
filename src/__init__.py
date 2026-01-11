"""
Agentic Insight Analyst - Source Package

A production-grade agentic system for synthesizing actionable insights
from qualitative and quantitative data.
"""

from src.agent import InsightAgent, AgentResult
from src.memory import WorkingMemory, ToolResult
from src.tools import tool_registry
from src.evaluator import Evaluator
from src.config import get_config, Config

__all__ = [
    "InsightAgent",
    "AgentResult",
    "WorkingMemory",
    "ToolResult",
    "tool_registry",
    "Evaluator",
    "get_config",
    "Config",
]

__version__ = "1.0.0"
