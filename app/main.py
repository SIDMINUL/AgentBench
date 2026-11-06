from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.config import settings
from app.engine import run
from app.judge import judge_run
from app.storage import get_run, list_runs, save_run, update_run
from app.tasks import TASKS

app = FastAPI(
    title="AgentBench",
    version="1.0.0",
    description="A reproducible platform for evaluating hosted LLM agents.",
)


class RunRequest(BaseModel):
    provider: str = settings.default_provider
    model: str = settings.default_model


class CompareRequest(BaseModel):
    provider: str = settings.default_provider
    models: list[str] = Field(min_length=2, max_length=8)
    repetitions: int = Field(default=3, ge=1, le=5)


class JudgeRequest(BaseModel):
    judge_model: str = settings.default_judge_model


def _aggregate_category_metrics(runs):
    categories = {}
    for item in runs:
        for category, metric in item.get("category_metrics", {}).items():
            target = categories.setdefault(
                category, {"total_tasks": 0, "successful_tasks": 0}
            )
            target["total_tasks"] += metric.get("total_tasks", 0)
            target["successful_tasks"] += metric.get("successful_tasks", 0)
    for metric in categories.values():
        metric["task_success_rate"] = (
            metric["successful_tasks"] / metric["total_tasks"]
            if metric["total_tasks"] else 0.0
        )
    return categories


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "AgentBench",
        "default_provider": settings.default_provider,
        "default_model": settings.default_model,
    }


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
            model_runs = [
                save_run(run(request.provider, model))
                for _ in range(request.repetitions)
            ]
            rates = [item["task_success_rate"] for item in model_runs]
            independent_rates = [
                item.get("model_independent_task_success_rate", item["task_success_rate"])
                for item in model_runs
            ]
            latencies = [item["average_latency_ms"] for item in model_runs]
            total_tasks = sum(item["total_tasks"] for item in model_runs)
            successful_tasks = sum(item["successful_tasks"] for item in model_runs)
            independent_successes = sum(
                item.get("model_independent_successful_tasks", item["successful_tasks"])
                for item in model_runs
            )
            tool_calls = sum(item.get("tool_calls", 0) for item in model_runs)
            tool_errors = sum(item.get("tool_errors", 0) for item in model_runs)
            duplicate_calls = sum(
                item.get("duplicate_tool_calls", 0) for item in model_runs
            )
            engine_assisted_calls = sum(
                item.get("engine_assisted_tool_calls", 0) for item in model_runs
            )
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
                "model_independent_successful_tasks": independent_successes,
                "model_independent_task_success_rate": (
                    independent_successes / total_tasks if total_tasks else 0.0
                ),
                "mean_model_independent_task_success_rate": (
                    sum(independent_rates) / len(independent_rates)
                ),
                "model_independent_success_rate_min": min(independent_rates),
                "model_independent_success_rate_max": max(independent_rates),
                "engine_assisted_tool_calls": engine_assisted_calls,
                "average_latency_ms": sum(latencies) / len(latencies),
                "mean_latency_ms": sum(latencies) / len(latencies),
                "p95_latency_ms": max(item["p95_latency_ms"] for item in model_runs),
                "tool_calls": tool_calls,
                "tool_errors": tool_errors,
                "duplicate_tool_calls": duplicate_calls,
                "tool_call_success_rate": (
                    (tool_calls - tool_errors - duplicate_calls) / tool_calls
                    if tool_calls else 1.0
                ),
                "category_metrics": _aggregate_category_metrics(model_runs),
                "per_run": [{
                    "run_id": item["run_id"],
                    "task_success_rate": item["task_success_rate"],
                    "model_independent_task_success_rate": item.get(
                        "model_independent_task_success_rate",
                        item["task_success_rate"],
                    ),
                    "average_latency_ms": item["average_latency_ms"],
                } for item in model_runs],
            })
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Rank by autonomous performance, not by tasks completed with engine assistance.
    ranked = sorted(comparisons, key=lambda item: (
        -item["mean_model_independent_task_success_rate"],
        item["mean_latency_ms"],
    ))
    return {
        "provider": request.provider,
        "repetitions": request.repetitions,
        "comparison_count": len(comparisons),
        "total_runs": len(comparisons) * request.repetitions,
        "ranking": ranked,
        "results": comparisons,
        "ranking_rule": (
            "Higher mean model-independent task success rate first; lower mean "
            "latency breaks ties. Pipeline success and model-independent success "
            "are reported separately. Success-rate ranges show run-to-run variability."
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
