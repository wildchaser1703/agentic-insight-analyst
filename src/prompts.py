"""
Prompts Module - Centralized system prompts for the agentic system.
"""

from typing import List, Dict, Any
import json

SYSTEM_IDENTITY = """You are an Insight Analyst Agent — a systematic AI specialized in analyzing data to produce actionable insights for leadership. Be explicit about reasoning, select tools dynamically, and stop when goals are satisfied."""

PLANNING_PROMPT = """Given a user's analysis goal, create a step-by-step plan.

USER GOAL: {goal}
DATA CONTEXT: {data_context}
AVAILABLE TOOLS: {available_tools}

Respond in JSON:
{{"interpreted_goal": "...", "success_criteria": [...], "planned_steps": [...], "estimated_complexity": "low|medium|high"}}"""

TOOL_SELECTION_PROMPT = """Select the next tool based on current state.

GOAL: {goal}
PROGRESS: {progress_summary}
INSIGHTS: {insights}
TOOLS: {tool_descriptions}

Respond in JSON:
{{"selected_tool": "name", "reasoning": "why", "expected_outcome": "what we expect", "confidence": 0.0-1.0}}"""

SYNTHESIS_PROMPT = """Synthesize findings into a leadership-ready report.

GOAL: {goal}
INSIGHTS: {insights}
TOOL OUTPUTS: {tool_outputs}

Respond in JSON:
{{"executive_summary": "...", "key_findings": [...], "recommendations": [...], "limitations": [...]}}"""


def get_planning_prompt(goal: str, data_context: str, available_tools: List[str]) -> str:
    return PLANNING_PROMPT.format(goal=goal, data_context=data_context, available_tools=", ".join(available_tools))


def get_tool_selection_prompt(goal: str, progress_summary: str, insights: List[str], tool_descriptions: Dict[str, str]) -> str:
    return TOOL_SELECTION_PROMPT.format(
        goal=goal, progress_summary=progress_summary,
        insights="\n".join(insights) if insights else "None",
        tool_descriptions=json.dumps(tool_descriptions, indent=2)
    )


def get_synthesis_prompt(goal: str, insights: List[str], tool_outputs: List[Dict]) -> str:
    return SYNTHESIS_PROMPT.format(
        goal=goal, insights="\n".join(insights),
        tool_outputs=json.dumps(tool_outputs[:5], indent=2)
    )
