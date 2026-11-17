"""
FastAPI Interface - REST API for the Agentic Insight Analyst.
"""

import os
import uuid
import tempfile
import csv
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

from src.agent import InsightAgent
from src.config import get_config

app = FastAPI(
    title="Agentic Insight Analyst API",
    description="Local agent prototype for synthesizing insights from qualitative and quantitative data",
    version="1.0.0",
)

config = get_config()
if config.api.cors_allow_all:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

jobs: Dict[str, Dict[str, Any]] = {}


class AnalysisRequest(BaseModel):
    goal: str = Field(..., description="Natural language analysis goal", min_length=10)
    data_path: Optional[str] = Field(None, description="Path to data file (CSV)")
    raw_text: Optional[str] = Field(None, description="Raw text data for analysis")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "goal": "Identify customer pain points",
                "data_path": "data/sample/sample_survey.csv",
            }
        }
    )


class AnalysisResponse(BaseModel):
    success: bool
    goal: str
    executive_summary: str
    key_findings: List[Any]
    recommendations: List[Any]
    reasoning_trace: List[str]
    iterations: int
    insights: List[str]
    error: Optional[str] = None


class JobStatus(BaseModel):
    job_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@app.get("/")
async def root():
    return {
        "service": "Agentic Insight Analyst",
        "status": "healthy",
        "version": "1.0.0",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "llm_configured": bool(os.getenv("OPENAI_API_KEY")),
        "max_iterations": config.agent.max_iterations,
    }


def validate_request(request):
    if bool(request.data_path) == bool(request.raw_text and request.raw_text.strip()):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of data_path or nonempty raw_text",
        )
    if request.data_path and not Path(request.data_path).is_file():
        raise HTTPException(status_code=404, detail="Data file not found")


@contextmanager
def input_path(request):
    if request.data_path:
        yield request.data_path
    else:
        with tempfile.TemporaryDirectory(prefix="insight-") as directory:
            path = Path(directory) / "feedback.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["feedback_text"])
                writer.writerows(
                    [
                        [line.strip()]
                        for line in request.raw_text.splitlines()
                        if line.strip()
                    ]
                )
            yield str(path)


def run_request(request):
    with input_path(request) as path:
        return InsightAgent().run(goal=request.goal, data_path=path)


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(request: AnalysisRequest):
    validate_request(request)
    result = run_request(request)
    output = result.final_output
    return AnalysisResponse(
        success=result.success,
        goal=result.goal,
        executive_summary=(
            output.get("executive_summary", "")
            if isinstance(output.get("executive_summary", ""), str)
            else ""
        ),
        key_findings=(
            output.get("key_findings", [])
            if isinstance(output.get("key_findings", []), list)
            else []
        ),
        recommendations=(
            output.get("recommendations", [])
            if isinstance(output.get("recommendations", []), list)
            else []
        ),
        reasoning_trace=result.reasoning_trace,
        iterations=result.iterations,
        insights=result.insights,
        error=(
            None
            if result.success
            else output.get(
                "error", "Analysis did not pass output validation or a tool failed"
            )
        ),
    )


@app.post("/analyze/async")
async def analyze_async(request: AnalysisRequest, background_tasks: BackgroundTasks):
    validate_request(request)
    if len(jobs) >= 100:
        completed = next(
            (
                key
                for key, value in jobs.items()
                if value["status"] in ("completed", "failed")
            ),
            None,
        )
        if completed is None:
            raise HTTPException(status_code=503, detail="Job capacity reached")
        del jobs[completed]

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "result": None, "error": None}
    background_tasks.add_task(run_analysis_job, job_id, request)
    return {"job_id": job_id, "status": "pending"}


def run_analysis_job(job_id: str, request: AnalysisRequest):
    jobs[job_id]["status"] = "running"
    try:
        result = run_request(request)
        jobs[job_id]["status"] = "completed" if result.success else "failed"
        if not result.success:
            jobs[job_id]["error"] = result.final_output.get(
                "error", "Analysis validation failed"
            )
        jobs[job_id]["result"] = {
            "success": result.success,
            "goal": result.goal,
            "final_output": result.final_output,
            "reasoning_trace": result.reasoning_trace,
            "iterations": result.iterations,
            "insights": result.insights,
        }
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        result=job.get("result"),
        error=job.get("error"),
    )


@app.get("/tools")
async def list_tools():
    from src.tools import tool_registry

    return {
        "tools": [
            {"name": name, "description": desc}
            for name, desc in tool_registry.get_descriptions().items()
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.api.host, port=config.api.port)
