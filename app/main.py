from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from app.engine import run
from app.storage import get_run, list_runs, save_run
from app.tasks import TASKS

app = FastAPI(
    title="AgentBench",
    version="0.2.0",
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
        result = run(request.provider, request.model)
        return save_run(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.get("/api/benchmark/history")
def benchmark_history(limit: int = Query(default=50, ge=1, le=200)):
    return {"runs": list_runs(limit)}

@app.get("/api/benchmark/history/{run_id}")
def benchmark_run_detail(run_id: int):
    result = get_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Benchmark run not found")
    return result
