"""
Agent Module - Core agentic loop with planner and controller.
Implements: Goal → Plan → Act → Observe → Decide → Synthesize → Stop
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from openai import OpenAI

from src.config import get_config
from src.memory import WorkingMemory, ToolResult
from src.tools import tool_registry
from src.evaluator import Evaluator
from src.prompts import SYSTEM_IDENTITY, get_planning_prompt, get_tool_selection_prompt, get_synthesis_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Final result from agent execution."""
    success: bool
    goal: str
    final_output: Dict[str, Any]
    reasoning_trace: List[str]
    iterations: int
    insights: List[str]
    memory_dump: Dict[str, Any]


class InsightAgent:
    """Agentic Insight Analyst - Core agent implementation."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.memory = WorkingMemory()
        self.evaluator = Evaluator(
            max_iterations=self.config.agent.max_iterations,
            min_improvement_threshold=self.config.agent.min_improvement_threshold
        )
        self._client: Optional[OpenAI] = None
        self._success_criteria: List[str] = []
    
    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self.config.llm.validate()
            self._client = OpenAI(api_key=self.config.llm.api_key)
        return self._client
    
    def run(self, goal: str, data_path: str) -> AgentResult:
        """Execute the full agentic loop."""
        logger.info(f"Starting agent with goal: {goal}")
        
        self.memory.clear()
        self.evaluator.reset()
        
        try:
            plan = self._plan(goal, data_path)
            logger.info(f"Generated plan with {len(plan)} steps")
            
            iteration = 0
            while True:
                iteration += 1
                logger.info(f"=== Iteration {iteration} ===")
                
                tool_result = self._execute_iteration(goal, data_path)
                self._observe(tool_result, goal)
                
                eval_result = self.evaluator.evaluate(
                    memory=self.memory,
                    current_iteration=iteration,
                    success_criteria=self._success_criteria
                )
                
                self.memory.record_evaluation(
                    progress=eval_result.progress_percentage,
                    should_continue=eval_result.should_continue,
                    reasoning=eval_result.reasoning
                )
                
                logger.info(f"Evaluation: {eval_result.reasoning}")
                
                if not eval_result.should_continue:
                    break
            
            final_output = self._synthesize(goal)
            
            return AgentResult(
                success=True,
                goal=goal,
                final_output=final_output,
                reasoning_trace=self.memory.get_reasoning_trace(),
                iterations=iteration,
                insights=self.memory.insights,
                memory_dump=self.memory.to_dict()
            )
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return AgentResult(
                success=False,
                goal=goal,
                final_output={"error": str(e)},
                reasoning_trace=self.memory.get_reasoning_trace(),
                iterations=0,
                insights=[],
                memory_dump=self.memory.to_dict()
            )
    
    def _plan(self, goal: str, data_path: str) -> List[str]:
        """Generate an explicit analysis plan."""
        data_analysis = tool_registry.execute("analyze_data_structure", data_path=data_path)
        data_context = json.dumps(data_analysis.get("output", {}), indent=2)
        
        prompt = get_planning_prompt(goal=goal, data_context=data_context, available_tools=tool_registry.list_tools())
        response = self._call_llm(prompt)
        plan_data = self._parse_json_response(response)
        
        steps = plan_data.get("planned_steps", [
            "Analyze data structure",
            "Extract key themes from feedback",
            "Cluster similar responses",
            "Generate insights",
            "Create recommendations"
        ])
        
        self._success_criteria = plan_data.get("success_criteria", ["Identify main themes", "Provide actionable recommendations"])
        self.memory.set_plan(goal=goal, steps=steps)
        return steps
    
    def _execute_iteration(self, goal: str, data_path: str) -> ToolResult:
        """Execute one iteration of tool selection and execution."""
        tool_name, reasoning = self._select_tool(goal)
        self.memory.record_tool_selection(tool_name=tool_name, reasoning=reasoning)
        
        tool_args = self._prepare_tool_args(tool_name, data_path)
        result_dict = tool_registry.execute(tool_name, **tool_args)
        
        tool_result = ToolResult(
            tool_name=tool_name,
            input_args=tool_args,
            output=result_dict.get("output"),
            success=result_dict.get("success", False),
            error_message=result_dict.get("error"),
            execution_time_ms=result_dict.get("execution_time_ms", 0)
        )
        
        self.memory.record_tool_result(tool_result)
        return tool_result
    
    def _select_tool(self, goal: str) -> tuple:
        """Dynamically select the next tool to use."""
        progress_parts = []
        for result in self.memory.tool_results:
            status = "✓" if result.success else "✗"
            progress_parts.append(f"{status} {result.tool_name}")
        progress_summary = ", ".join(progress_parts) if progress_parts else "No tools executed yet"
        
        prompt = get_tool_selection_prompt(
            goal=goal,
            progress_summary=progress_summary,
            insights=self.memory.insights,
            tool_descriptions=tool_registry.get_descriptions()
        )
        
        response = self._call_llm(prompt)
        selection = self._parse_json_response(response)
        
        tool_name = selection.get("selected_tool", "analyze_data_structure")
        reasoning = selection.get("reasoning", "Default selection")
        
        if tool_name not in tool_registry.list_tools():
            tool_name = "analyze_data_structure"
            reasoning = "Fallback to default tool"
        
        return tool_name, reasoning
    
    def _prepare_tool_args(self, tool_name: str, data_path: str) -> Dict[str, Any]:
        """Prepare arguments for a tool based on current context."""
        import pandas as pd
        
        if tool_name == "analyze_data_structure":
            return {"data_path": data_path}
        
        elif tool_name == "analyze_sentiment_distribution":
            data_result = self.memory.get_tool_results_by_name("analyze_data_structure")
            if data_result and data_result[0].success:
                score_cols = data_result[0].output.get("likely_likert_columns", [])
                return {"data_path": data_path, "score_columns": score_cols}
            return {"data_path": data_path, "score_columns": []}
        
        elif tool_name in ["summarize_text", "cluster_feedback", "extract_key_phrases"]:
            df = pd.read_csv(data_path)
            data_result = self.memory.get_tool_results_by_name("analyze_data_structure")
            text_col = "feedback_text"
            if data_result and data_result[0].success:
                feedback_cols = data_result[0].output.get("likely_feedback_columns", [])
                if feedback_cols:
                    text_col = feedback_cols[0]
            
            if text_col in df.columns:
                texts = df[text_col].dropna().tolist()
            else:
                texts = df.select_dtypes(include=['object']).iloc[:, -1].dropna().tolist()
            return {"texts": texts}
        
        elif tool_name == "generate_recommendations":
            return {"insights": self.memory.insights}
        
        return {}
    
    def _observe(self, tool_result: ToolResult, goal: str) -> None:
        """Extract observations and insights from tool result."""
        if not tool_result.success:
            self.memory.record_observation(f"Tool {tool_result.tool_name} failed: {tool_result.error_message}")
            return
        
        output = tool_result.output
        
        if tool_result.tool_name == "cluster_feedback" and isinstance(output, dict):
            clusters = output.get("clusters", {})
            for cluster_id, cluster_data in clusters.items():
                if cluster_data.get("top_terms"):
                    terms = ", ".join(cluster_data["top_terms"][:3])
                    insight = f"Theme identified: {terms} ({cluster_data.get('count', 0)} responses)"
                    self.memory.add_insight(insight)
        
        elif tool_result.tool_name == "analyze_sentiment_distribution" and isinstance(output, dict):
            if "overall" in output:
                overall = output["overall"]
                insight = f"Overall sentiment: {overall.get('interpretation', 'unknown')} (avg: {overall.get('average_across_metrics', 0):.2f})"
                self.memory.add_insight(insight)
        
        elif tool_result.tool_name == "extract_key_phrases" and isinstance(output, list):
            if output:
                top_phrases = [p["phrase"] for p in output[:5]]
                insight = f"Top concerns: {', '.join(top_phrases)}"
                self.memory.add_insight(insight)
        
        self.memory.record_observation(f"Processed {tool_result.tool_name} output successfully", data={"output_type": type(output).__name__})
    
    def _synthesize(self, goal: str) -> Dict[str, Any]:
        """Synthesize all findings into final output."""
        prompt = get_synthesis_prompt(goal=goal, insights=self.memory.insights, tool_outputs=[r.to_dict() for r in self.memory.tool_results])
        response = self._call_llm(prompt)
        synthesis = self._parse_json_response(response)
        
        if "executive_summary" not in synthesis:
            synthesis["executive_summary"] = f"Analysis of goal: {goal}"
        if "key_findings" not in synthesis:
            synthesis["key_findings"] = self.memory.insights
        if "recommendations" not in synthesis:
            synthesis["recommendations"] = ["Review identified themes for action items"]
        
        validation = self.evaluator.validate_output(synthesis, goal)
        synthesis["_validation"] = validation
        return synthesis
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.llm.model,
                messages=[
                    {"role": "system", "content": SYSTEM_IDENTITY},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"LLM call failed: {e}, using fallback")
            return "{}"
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            return {}
