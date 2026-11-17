from pathlib import Path
from unittest.mock import Mock
import pytest
from fastapi.testclient import TestClient
from src.agent import InsightAgent
from src.config import Config
from api import main

DATA = str(Path(__file__).parents[1] / "data/sample/sample_survey.csv")


def agent_with(*responses):
    config = Config()
    config.agent.max_iterations = 1
    agent = InsightAgent(config)
    agent._call_llm = Mock(side_effect=responses)
    return agent


def test_provider_failure_is_not_success():
    agent = agent_with(RuntimeError("provider unavailable"))
    result = agent.run("Find feedback themes", DATA)
    assert not result.success
    assert result.iterations == 0
    assert "provider unavailable" in result.final_output["error"]


@pytest.mark.parametrize("response", ["{}", "[]", "garbage", "null"])
def test_invalid_model_output_fails(response):
    assert not agent_with(response).run("Find themes", DATA).success


def test_valid_flow_and_invalid_synthesis():
    plan = '{"planned_steps":["Inspect input"]}'
    selection = '{"selected_tool":"analyze_data_structure"}'
    good = '{"executive_summary":"Feedback inspected","key_findings":["50 responses","Includes scores"],"recommendations":["Review feedback"]}'
    assert agent_with(plan, selection, good).run("Inspect feedback", DATA).success
    result = agent_with(plan, selection, '{"executive_summary":"Only summary"}').run(
        "Inspect feedback", DATA
    )
    assert not result.success and result.iterations == 1


def test_raw_text_is_consumed_and_deleted(monkeypatch):
    observed = []
    from src.agent import AgentResult

    def fake_run(self, goal, data_path):
        import pandas as pd

        observed.append(data_path)
        assert pd.read_csv(data_path).feedback_text.tolist() == [
            "Salty yet sweet",
            "Slow support",
        ]
        return AgentResult(False, goal, {"error": "test failure"}, [], 0, [], {})

    monkeypatch.setattr(main.InsightAgent, "run", fake_run)
    client = TestClient(main.app)
    response = client.post(
        "/analyze",
        json={
            "goal": "Find the feedback themes",
            "raw_text": "Salty yet sweet\nSlow support",
        },
    )
    assert response.status_code == 200
    assert response.json()["error"] == "test failure"
    assert not Path(observed[0]).exists()
    response = client.post(
        "/analyze/async",
        json={
            "goal": "Find feedback themes",
            "raw_text": "Salty yet sweet\nSlow support",
        },
    )
    status = client.get("/jobs/" + response.json()["job_id"]).json()
    assert status["status"] == "failed"
    assert not Path(observed[-1]).exists()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"raw_text": "  "},
        {"data_path": "missing.csv"},
        {"data_path": DATA, "raw_text": "hello"},
    ],
)
def test_api_rejects_invalid_input(payload):
    client = TestClient(main.app)
    for route in ["/analyze", "/analyze/async"]:
        assert client.post(
            route, json={"goal": "Find feedback themes", **payload}
        ).status_code in (400, 404)


def test_unknown_tool_does_not_silently_fall_back():
    result = agent_with(
        '{"planned_steps":["Inspect input"]}', '{"selected_tool":"invented_tool"}'
    ).run("Inspect feedback", DATA)
    assert not result.success and "unknown tool" in result.final_output["error"]
