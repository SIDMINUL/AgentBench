import math
import re
import time
from statistics import mean

from app.providers import get_provider
from app.tasks import TASKS

def normalize(text: str) -> str:
    return re.sub(r"\\s+", " ", text).strip().casefold().strip(" .")

def run(provider_name: str = "mock", model: str = "mock-agent") -> dict:
    provider = get_provider(provider_name)
    results = []

    for task in TASKS:
        started = time.perf_counter()
        error = None
        try:
            output = provider.answer(task["prompt"], model)
            success = normalize(output) == normalize(task["expected"])
        except Exception as exc:
            output = ""
            success = False
            error = str(exc)

        results.append({
            "task_id": task["id"],
            "category": task["category"],
            "output": output,
            "expected": task["expected"],
            "success": success,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "error": error,
        })

    latencies = sorted(item["latency_ms"] for item in results)
    total = len(results)
    passed = sum(item["success"] for item in results)
    p95 = latencies[max(0, math.ceil(0.95 * total) - 1)] if total else 0.0

    return {
        "provider": provider_name,
        "model": model,
        "total_tasks": total,
        "successful_tasks": passed,
        "task_success_rate": passed / total if total else 0.0,
        "average_latency_ms": mean(latencies) if latencies else 0.0,
        "p95_latency_ms": p95,
        "results": results,
    }
