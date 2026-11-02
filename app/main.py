from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.engine import run
from app.tasks import TASKS

app = FastAPI(
    title="AgentBench",
    version="0.1.0",
    description="A reproducible benchmark platform for LLM agents.",
)

class RunRequest(BaseModel):
    provider: str = "mock"
    model: str = "mock-agent"

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "AgentBench"}

@app.get("/api/tasks")
def list_tasks():
    return TASKS

@app.post("/api/benchmark/run")
def run_benchmark(request: RunRequest):
    try:
        return run(request.provider, request.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
