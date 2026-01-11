"""
Integration Tests - Test the complete agent flow.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import InsightAgent, AgentResult
from src.memory import WorkingMemory, ToolResult, StepType
from src.tools import tool_registry, analyze_data_structure, cluster_feedback
from src.evaluator import Evaluator, EvaluationResult
from src.config import Config, LLMConfig, AgentConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_data_path():
    return str(Path(__file__).parent.parent / "data" / "sample" / "sample_survey.csv")


@pytest.fixture
def mock_config():
    config = Config()
    config.llm.api_key = "test-key"
    config.agent.max_iterations = 5
    return config


@pytest.fixture
def memory():
    return WorkingMemory()


@pytest.fixture
def evaluator():
    return Evaluator(max_iterations=5)


# =============================================================================
# Memory Tests
# =============================================================================

class TestWorkingMemory:
    def test_set_plan(self, memory):
        plan = memory.set_plan(goal="Test goal", steps=["Step 1", "Step 2"])
        assert plan.goal == "Test goal"
        assert len(plan.steps) == 2
        assert memory.step_count == 1
    
    def test_record_tool_result(self, memory):
        result = ToolResult(
            tool_name="test_tool",
            input_args={"arg": "value"},
            output={"result": "data"},
            success=True
        )
        memory.record_tool_result(result)
        assert len(memory.tool_results) == 1
        assert memory.tool_results[0].tool_name == "test_tool"
    
    def test_add_insight(self, memory):
        memory.add_insight("Test insight 1")
        memory.add_insight("Test insight 2")
        assert len(memory.insights) == 2
    
    def test_get_reasoning_trace(self, memory):
        memory.set_plan(goal="Test", steps=["Step 1"])
        memory.record_observation("Observation 1")
        trace = memory.get_reasoning_trace()
        assert len(trace) == 2
    
    def test_clear(self, memory):
        memory.set_plan(goal="Test", steps=["Step 1"])
        memory.add_insight("Insight")
        memory.clear()
        assert memory.plan is None
        assert len(memory.insights) == 0
        assert memory.step_count == 0
    
    def test_to_dict(self, memory):
        memory.set_plan(goal="Test", steps=["Step 1"])
        data = memory.to_dict()
        assert "plan" in data
        assert "entries" in data
        assert "insights" in data


# =============================================================================
# Tool Tests
# =============================================================================

class TestTools:
    def test_tool_registry_list(self):
        tools = tool_registry.list_tools()
        assert "analyze_data_structure" in tools
        assert "cluster_feedback" in tools
        assert "summarize_text" in tools
        assert "generate_recommendations" in tools
    
    def test_tool_descriptions(self):
        descriptions = tool_registry.get_descriptions()
        assert len(descriptions) >= 4
        assert all(isinstance(v, str) for v in descriptions.values())
    
    def test_analyze_data_structure(self, sample_data_path):
        result = analyze_data_structure(sample_data_path)
        assert "row_count" in result
        assert "column_count" in result
        assert "columns" in result
        assert result["row_count"] == 50
    
    def test_cluster_feedback(self):
        texts = [
            "The pricing is too high for what we get",
            "Pricing needs to be more competitive",
            "Support is slow to respond",
            "Customer support takes forever",
            "Great product overall",
            "Love the features",
            "The app crashes frequently",
            "Too many bugs in the system"
        ]
        result = cluster_feedback(texts, n_clusters=2)
        assert "clusters" in result
        assert result["total_texts"] == 8
    
    def test_tool_execution_success(self, sample_data_path):
        result = tool_registry.execute("analyze_data_structure", data_path=sample_data_path)
        assert result["success"] is True
        assert result["output"] is not None
    
    def test_tool_execution_not_found(self):
        result = tool_registry.execute("nonexistent_tool")
        assert result["success"] is False
        assert "not found" in result["error"]


# =============================================================================
# Evaluator Tests
# =============================================================================

class TestEvaluator:
    def test_max_iterations_stop(self, evaluator, memory):
        result = evaluator.evaluate(memory, current_iteration=10, success_criteria=[])
        assert result.should_continue is False
        assert "Maximum iterations" in result.reasoning
    
    def test_continue_early(self, evaluator, memory):
        result = evaluator.evaluate(memory, current_iteration=1, success_criteria=[])
        assert result.should_continue is True
    
    def test_progress_estimation(self, evaluator, memory):
        memory.add_insight("Insight 1")
        memory.add_insight("Insight 2")
        memory.record_tool_result(ToolResult(
            tool_name="test", input_args={}, output={}, success=True
        ))
        result = evaluator.evaluate(memory, current_iteration=1, success_criteria=[])
        assert result.progress_percentage > 0
    
    def test_validate_output_valid(self, evaluator):
        output = {
            "executive_summary": "Summary here",
            "key_findings": ["Finding 1", "Finding 2"],
            "recommendations": ["Rec 1"]
        }
        validation = evaluator.validate_output(output, "Test goal")
        assert validation["valid"] is True
    
    def test_validate_output_missing_fields(self, evaluator):
        output = {"executive_summary": "Summary here"}
        validation = evaluator.validate_output(output, "Test goal")
        assert validation["valid"] is False
        assert len(validation["issues"]) > 0
    
    def test_reset(self, evaluator, memory):
        evaluator.evaluate(memory, current_iteration=1, success_criteria=[])
        evaluator.reset()
        assert evaluator._previous_progress == 0.0
        assert evaluator._no_progress_count == 0


# =============================================================================
# Agent Tests (with mocked LLM)
# =============================================================================

class TestInsightAgent:
    @patch('src.agent.OpenAI')
    def test_agent_initialization(self, mock_openai, mock_config):
        agent = InsightAgent(config=mock_config)
        assert agent.memory is not None
        assert agent.evaluator is not None
    
    @patch('src.agent.OpenAI')
    def test_agent_run_with_mock_llm(self, mock_openai, mock_config, sample_data_path):
        # Mock LLM responses
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"planned_steps": ["Step 1"], "success_criteria": ["Criterion 1"]}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        agent = InsightAgent(config=mock_config)
        agent._client = mock_client
        
        result = agent.run(goal="Test analysis goal", data_path=sample_data_path)
        
        assert isinstance(result, AgentResult)
        assert result.goal == "Test analysis goal"
        assert result.memory_dump is not None
    
    def test_parse_json_response(self, mock_config):
        agent = InsightAgent(config=mock_config)
        
        # Test normal JSON
        result = agent._parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}
        
        # Test markdown code block
        result = agent._parse_json_response('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}
        
        # Test invalid JSON
        result = agent._parse_json_response('not json')
        assert result == {}


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    def test_complete_analysis_flow_with_tools(self, sample_data_path):
        """Test the complete flow without LLM (using fallbacks)."""
        # Test data structure analysis
        data_result = tool_registry.execute("analyze_data_structure", data_path=sample_data_path)
        assert data_result["success"]
        
        # Get feedback texts
        import pandas as pd
        df = pd.read_csv(sample_data_path)
        texts = df["feedback_text"].dropna().tolist()
        
        # Test clustering
        cluster_result = tool_registry.execute("cluster_feedback", texts=texts)
        assert cluster_result["success"]
        assert "clusters" in cluster_result["output"]
        
        # Test key phrase extraction
        phrase_result = tool_registry.execute("extract_key_phrases", texts=texts)
        assert phrase_result["success"]
        
        # Test sentiment analysis
        sentiment_result = tool_registry.execute(
            "analyze_sentiment_distribution",
            data_path=sample_data_path,
            score_columns=["satisfaction_score", "ease_of_use"]
        )
        assert sentiment_result["success"]
        assert "overall" in sentiment_result["output"]
    
    def test_memory_and_evaluator_integration(self, memory, evaluator):
        """Test memory and evaluator working together."""
        memory.set_plan(goal="Test goal", steps=["Step 1", "Step 2"])
        
        for i in range(3):
            memory.add_insight(f"Insight {i}")
            memory.record_tool_result(ToolResult(
                tool_name=f"tool_{i}",
                input_args={},
                output={},
                success=True
            ))
        
        result = evaluator.evaluate(memory, current_iteration=3, success_criteria=[])
        assert result.progress_percentage > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
