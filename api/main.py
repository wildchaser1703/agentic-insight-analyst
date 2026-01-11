"""
FastAPI Interface - REST API for the Agentic Insight Analyst.
"""

import os
import uuid
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent import InsightAgent
from src.config import get_config

app = FastAPI(
    title="Agentic Insight Analyst API",
    description="Production-grade agentic system for synthesizing insights from qualitative and quantitative data",
    version="1.0.0"
)

config = get_config()
if config.api.cors_allow_all:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

jobs: Dict[str, Dict[str, Any]] = {}


class AnalysisRequest(BaseModel):
    goal: str = Field(..., description="Natural language analysis goal", min_length=10)
    data_path: Optional[str] = Field(None, description="Path to data file (CSV)")
    raw_text: Optional[str] = Field(None, description="Raw text data for analysis")
    
    class Config:
        json_schema_extra = {
            "example": {
                "goal": "Identify the top 3 customer pain points and recommend improvements",
                "data_path": "data/sample/sample_survey.csv"
            }
        }


class AnalysisResponse(BaseModel):
    success: bool
    goal: str
    executive_summary: str
    key_findings: List[Any]
    recommendations: List[Any]
    reasoning_trace: List[str]
    iterations: int
    insights: List[str]


class JobStatus(BaseModel):
    job_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@app.get("/")
async def root():
    return {"service": "Agentic Insight Analyst", "status": "healthy", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "llm_configured": bool(os.getenv("OPENAI_API_KEY")),
        "max_iterations": config.agent.max_iterations
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    if not request.data_path and not request.raw_text:
        raise HTTPException(status_code=400, detail="Either data_path or raw_text must be provided")
    
    if request.data_path and not os.path.exists(request.data_path):
        raise HTTPException(status_code=404, detail=f"Data file not found: {request.data_path}")
    
    try:
        agent = InsightAgent()
        result = agent.run(goal=request.goal, data_path=request.data_path or "")
        
        return AnalysisResponse(
            success=result.success,
            goal=result.goal,
            executive_summary=result.final_output.get("executive_summary", ""),
            key_findings=result.final_output.get("key_findings", []),
            recommendations=result.final_output.get("recommendations", []),
            reasoning_trace=result.reasoning_trace,
            iterations=result.iterations,
            insights=result.insights
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/async")
async def analyze_async(request: AnalysisRequest, background_tasks: BackgroundTasks):
    if not request.data_path and not request.raw_text:
        raise HTTPException(status_code=400, detail="Either data_path or raw_text must be provided")
    
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "result": None, "error": None}
    background_tasks.add_task(run_analysis_job, job_id, request)
    return {"job_id": job_id, "status": "pending"}


def run_analysis_job(job_id: str, request: AnalysisRequest):
    jobs[job_id]["status"] = "running"
    try:
        agent = InsightAgent()
        result = agent.run(goal=request.goal, data_path=request.data_path or "")
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = {
            "success": result.success,
            "goal": result.goal,
            "final_output": result.final_output,
            "reasoning_trace": result.reasoning_trace,
            "iterations": result.iterations,
            "insights": result.insights
        }
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    return JobStatus(job_id=job_id, status=job["status"], result=job.get("result"), error=job.get("error"))


@app.get("/tools")
async def list_tools():
    from src.tools import tool_registry
    return {"tools": [{"name": name, "description": desc} for name, desc in tool_registry.get_descriptions().items()]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.api.host, port=config.api.port)
