from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.engine import run
from app.judge import judge_run
from app.storage import get_run, list_runs, save_run, update_run
from app.tasks import TASKS

app = FastAPI(title="AgentBench", version="0.7.0",
              description="A reproducible benchmark platform for LLM agents.")


class RunRequest(BaseModel):
    provider: str = "mock"
    model: str = "mock-agent"


class CompareRequest(BaseModel):
    provider: str = "ollama"
    models: list[str] = Field(min_length=2, max_length=8)
    repetitions: int = Field(default=1, ge=1, le=5)


class JudgeRequest(BaseModel):
    judge_model: str = "llama3.2:3b"


def _aggregate_category_metrics(runs):
    categories = {}
    for item in runs:
        for category, metric in item.get("category_metrics", {}).items():
            target = categories.setdefault(category, {"total_tasks": 0, "successful_tasks": 0})
            target["total_tasks"] += metric.get("total_tasks", 0)
            target["successful_tasks"] += metric.get("successful_tasks", 0)
    for metric in categories.values():
        metric["task_success_rate"] = (
            metric["successful_tasks"] / metric["total_tasks"] if metric["total_tasks"] else 0.0
        )
    return categories


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "AgentBench"}


@app.get("/api/tasks")
def list_tasks():
    return TASKS


@app.post("/api/benchmark/run")
def run_benchmark(request: RunRequest):
    try:
        return save_run(run(request.provider, request.model))
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
            model_runs = [save_run(run(request.provider, model))
                          for _ in range(request.repetitions)]
            rates = [item["task_success_rate"] for item in model_runs]
            latencies = [item["average_latency_ms"] for item in model_runs]
            total_tasks = sum(item["total_tasks"] for item in model_runs)
            successful_tasks = sum(item["successful_tasks"] for item in model_runs)
            comparisons.append({
                "model": model,
                "run_ids": [item["run_id"] for item in model_runs],
                "repetitions": len(model_runs),
                "task_success_rate": sum(rates) / len(rates),
                "mean_task_success_rate": sum(rates) / len(rates),
                "success_rate_min": min(rates),
                "success_rate_max": max(rates),
                "successful_tasks": successful_tasks,
                "total_tasks": total_tasks,
                "average_latency_ms": sum(latencies) / len(latencies),
                "mean_latency_ms": sum(latencies) / len(latencies),
                "p95_latency_ms": max(item["p95_latency_ms"] for item in model_runs),
                "tool_calls": sum(item["tool_calls"] for item in model_runs),
                "tool_errors": sum(item["tool_errors"] for item in model_runs),
                "tool_call_success_rate": (
                    sum(item["tool_calls"] - item["tool_errors"] for item in model_runs)
                    / sum(item["tool_calls"] for item in model_runs)
                    if sum(item["tool_calls"] for item in model_runs) else 1.0
                ),
                "category_metrics": _aggregate_category_metrics(model_runs),
                "per_run": [{
                    "run_id": item["run_id"],
                    "task_success_rate": item["task_success_rate"],
                    "average_latency_ms": item["average_latency_ms"],
                } for item in model_runs],
            })
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    ranked = sorted(comparisons, key=lambda item: (
        -item["mean_task_success_rate"], item["mean_latency_ms"]
    ))
    return {
        "provider": request.provider,
        "repetitions": request.repetitions,
        "comparison_count": len(comparisons),
        "total_runs": len(comparisons) * request.repetitions,
        "ranking": ranked,
        "results": comparisons,
        "ranking_rule": (
            "Higher mean task success rate first; lower mean latency breaks ties. "
            "Success-rate range shows run-to-run variability."
        ),
    }


@app.post("/api/benchmark/history/{run_id}/judge")
def judge_benchmark_run(run_id: int, request: JudgeRequest):
    result = get_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Benchmark run not found")
    try:
        evaluation = judge_run(result, request.judge_model)
        result["llm_judge"] = evaluation
        update_run(run_id, result)
        return {"run_id": run_id, **evaluation}
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/api/benchmark/history")
def benchmark_history(limit: int = Query(default=50, ge=1, le=200)):
    return {"runs": list_runs(limit)}


@app.get("/api/benchmark/history/{run_id}")
def benchmark_run_detail(run_id: int):
    result = get_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Benchmark run not found")
    return result
