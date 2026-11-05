import json
import math
import re
import time
from statistics import mean

from app.providers import get_provider
from app.tasks import TASKS
from app.tools import execute_tool

def normalize(text: str) -> str:
    text = re.sub(r"\s*,\s*", ", ", text)
    return re.sub(r"\s+", " ", text).strip().casefold().strip(" .")

def run_task(task: dict, provider, provider_name: str, model: str) -> dict:
    started = time.perf_counter()
    history = []
    trace = []
    tool_calls = 0
    tool_errors = 0
    duplicate_tool_calls = 0
    tool_call_cache = {}
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
                recovery = task.get("required_recovery")
                has_tool_error = any(
                    item.get("type") == "tool_error" for item in history
                )
                recovery_completed = False
                if recovery:
                    required_tool = recovery["tool"]
                    required_arguments = recovery["arguments"]
                    recovery_completed = any(
                        item.get("type") == "tool_result"
                        and item.get("tool") == required_tool
                        and all(
                            item.get("result", {}).get(key) == value
                            for key, value in required_arguments.items()
                        )
                        for item in history
                    )
                if recovery and has_tool_error and not recovery_completed:
                    message = (
                        "Final answer blocked: the required recovery lookup has not "
                        "succeeded. Execute the task's REQUIRED RECOVERY ACTION before "
                        "returning a final answer."
                    )
                    history.append({
                        "type": "recovery_required",
                        "tool": recovery["tool"],
                        "arguments": recovery["arguments"],
                        "message": message,
                    })
                    trace.append({
                        "step": step_number,
                        "type": "final_blocked_recovery",
                        "answer": str(action.get("answer", "")),
                        "message": message,
                    })
                    status = "running"
                    continue

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

            # Cache both successful and failed executions. Repeating an identical
            # call cannot change a deterministic tool result and often signals a loop.
            signature = (
                tool_name,
                json.dumps(arguments, sort_keys=True, separators=(",", ":")),
            )
            if signature in tool_call_cache:
                duplicate_tool_calls += 1
                cached = tool_call_cache[signature]
                message = (
                    "Repeated tool call suppressed. This exact tool call was already "
                    "executed. Use the cached observation in history and choose a "
                    "different next action or return a final answer."
                )
                duplicate_event = {
                    "type": "duplicate_tool_call",
                    "tool": tool_name,
                    "arguments": arguments,
                    "cached_success": cached["success"],
                    "message": message,
                }
                if cached["success"]:
                    duplicate_event["previous_result"] = cached["result"]
                else:
                    duplicate_event["previous_error"] = cached["error"]
                history.append(duplicate_event)

                # Replay the cached observation in the same shape as a normal
                # tool response. Smaller models often fail to interpret custom
                # duplicate-event metadata, but reliably consume tool_result /
                # tool_error observations.
                if cached["success"]:
                    history.append({
                        "type": "tool_result",
                        "tool": tool_name,
                        "result": cached["result"],
                        "replayed": True,
                    })
                else:
                    history.append({
                        "type": "tool_error",
                        "tool": tool_name,
                        "error": cached["error"],
                        "replayed": True,
                    })

                trace_event = {
                    "step": step_number,
                    "type": "tool_call_duplicate",
                    "tool": tool_name,
                    "arguments": arguments,
                    "success": False,
                    "message": message,
                }
                if cached["success"]:
                    trace_event["previous_result"] = cached["result"]
                else:
                    trace_event["previous_error"] = cached["error"]
                trace.append(trace_event)

                # One suppressed repeat gives the model a chance to recover.
                # A second repeat ends the task rather than wasting more steps.
                if duplicate_tool_calls >= 2:
                    status = "repeated_action"
                    error = (
                        "Agent repeated an already-executed tool call twice; "
                        "execution stopped to prevent an infinite action loop."
                    )
                    trace[-1]["message"] = error
                    break
                status = "running"
                continue

            try:
                result = execute_tool(tool_name, arguments)
                tool_call_cache[signature] = {"success": True, "result": result}
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
                tool_call_cache[signature] = {"success": False, "error": str(tool_exc)}
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
        "duplicate_tool_calls": duplicate_tool_calls,
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
    duplicate_tool_calls = sum(item["duplicate_tool_calls"] for item in results)
    p95 = latencies[max(0, math.ceil(0.95 * total) - 1)] if total else 0.0

    category_metrics = {}
    for category in sorted({item["category"] for item in results}):
        category_results = [
            item for item in results if item["category"] == category
        ]
        category_total = len(category_results)
        category_passed = sum(item["success"] for item in category_results)
        category_metrics[category] = {
            "total_tasks": category_total,
            "successful_tasks": category_passed,
            "task_success_rate": (
                category_passed / category_total if category_total else 0.0
            ),
        }

    return {
        "provider": provider_name,
        "model": model,
        "category_metrics": category_metrics,
        "total_tasks": total,
        "successful_tasks": passed,
        "task_success_rate": passed / total if total else 0.0,
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "duplicate_tool_calls": duplicate_tool_calls,
        "tool_call_success_rate": (
            tool_calls - tool_errors - duplicate_tool_calls
        ) / tool_calls if tool_calls else 1.0,
        "average_latency_ms": mean(latencies) if latencies else 0.0,
        "p95_latency_ms": p95,
        "results": results,
    }
