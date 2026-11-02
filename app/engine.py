import math
import re
import time
from statistics import mean

from app.providers import get_provider
from app.tasks import TASKS
from app.tools import execute_tool

def normalize(text: str) -> str:
    return re.sub(r"\\s+", " ", text).strip().casefold().strip(" .")

def run_task(task: dict, provider, provider_name: str, model: str) -> dict:
    started = time.perf_counter()
    history = []
    trace = []
    tool_calls = 0
    tool_errors = 0
    output = ""
    error = None
    success = False
    status = "step_limit"

    for step_number in range(1, task.get("max_steps", 4) + 1):
        try:
            action = provider.next_action(task, history, model)
            if not isinstance(action, dict):
                raise ValueError("Agent action must be a JSON object")

            if action.get("type") == "final":
                output = str(action.get("answer", ""))
                success = normalize(output) == normalize(task["expected"])
                status = "completed"
                trace.append({
                    "step": step_number,
                    "type": "final",
                    "answer": output,
                    "success": success,
                })
                break

            if action.get("type") != "tool_call":
                raise ValueError("Unknown action type")
            tool_name = action.get("tool")
            if tool_name not in task.get("tools", []):
                raise ValueError(f"Tool not allowed for this task: {tool_name}")

            tool_calls += 1
            arguments = action.get("arguments", {})
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be an object")
            try:
                result = execute_tool(tool_name, arguments)
                history.append({"type": "tool_result", "tool": tool_name, "result": result})
                trace.append({
                    "step": step_number,
                    "type": "tool_call",
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result,
                    "success": True,
                })
            except Exception as tool_exc:
                tool_errors += 1
                history.append({"type": "tool_error", "tool": tool_name, "error": str(tool_exc)})
                trace.append({
                    "step": step_number,
                    "type": "tool_call",
                    "tool": tool_name,
                    "arguments": arguments,
                    "error": str(tool_exc),
                    "success": False,
                })
        except Exception as exc:
            error = str(exc)
            status = "error"
            trace.append({"step": step_number, "type": "error", "error": error})
            break

    latency_ms = (time.perf_counter() - started) * 1000
    return {
        "task_id": task["id"],
        "category": task["category"],
        "output": output,
        "expected": task["expected"],
        "success": success,
        "status": status,
        "latency_ms": latency_ms,
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "error": error,
        "trace": trace,
    }

def run(provider_name: str = "mock", model: str = "mock-agent") -> dict:
    provider = get_provider(provider_name)
    results = [run_task(task, provider, provider_name, model) for task in TASKS]
    latencies = sorted(item["latency_ms"] for item in results)
    total = len(results)
    passed = sum(item["success"] for item in results)
    tool_calls = sum(item["tool_calls"] for item in results)
    tool_errors = sum(item["tool_errors"] for item in results)
    p95 = latencies[max(0, math.ceil(0.95 * total) - 1)] if total else 0.0

    return {
        "provider": provider_name,
        "model": model,
        "total_tasks": total,
        "successful_tasks": passed,
        "task_success_rate": passed / total if total else 0.0,
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "tool_call_success_rate": (tool_calls - tool_errors) / tool_calls if tool_calls else 1.0,
        "average_latency_ms": mean(latencies) if latencies else 0.0,
        "p95_latency_ms": p95,
        "results": results,
    }
