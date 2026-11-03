from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.engine import run
from app.storage import get_run, list_runs, save_run
from app.tasks import TASKS

app = FastAPI(
    title="AgentBench",
    version="0.3.0",
    description="A reproducible benchmark platform for LLM agents.",
)


class RunRequest(BaseModel):
    provider: str = "mock"
    model: str = "mock-agent"


class CompareRequest(BaseModel):
    provider: str = "ollama"
    models: list[str] = Field(min_length=2, max_length=8)


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


@app.post("/api/benchmark/compare")
def compare_models(request: CompareRequest):
    models = [model.strip() for model in request.models]
    if any(not model for model in models):
        raise HTTPException(status_code=400, detail="Model names cannot be empty")
    if len({model.casefold() for model in models}) != len(models):
        raise HTTPException(status_code=400, detail="Model names must be unique")

    comparisons = []
    try:
        for model in models:
            result = save_run(run(request.provider, model))
            comparisons.append({
                "run_id": result["run_id"],
                "model": result["model"],
                "task_success_rate": result["task_success_rate"],
                "successful_tasks": result["successful_tasks"],
                "total_tasks": result["total_tasks"],
                "average_latency_ms": result["average_latency_ms"],
                "p95_latency_ms": result["p95_latency_ms"],
                "tool_calls": result["tool_calls"],
                "tool_errors": result["tool_errors"],
                "tool_call_success_rate": result["tool_call_success_rate"],
            })
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    ranked = sorted(
        comparisons,
        key=lambda item: (-item["task_success_rate"], item["average_latency_ms"]),
    )
    return {
        "provider": request.provider,
        "comparison_count": len(comparisons),
        "ranking": ranked,
        "results": comparisons,
        "ranking_rule": "Higher task success rate first; lower average latency breaks ties.",
    }


@app.get("/api/benchmark/history")
def benchmark_history(limit: int = Query(default=50, ge=1, le=200)):
    return {"runs": list_runs(limit)}


@app.get("/api/benchmark/history/{run_id}")
def benchmark_run_detail(run_id: int):
    result = get_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Benchmark run not found")
    return result
