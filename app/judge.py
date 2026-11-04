import json
import httpx

from app.config import settings
from app.tasks import TASKS


def _deterministic_tool_score(task: dict, trace: list) -> int:
    required_tools = set(task.get("tools", []))
    calls = [
        step for step in trace
        if step.get("type") == "tool_call"
    ]

    if not required_tools:
        return 5 if not calls else 1

    successful_tools = {
        step.get("tool")
        for step in calls
        if step.get("success") is True
    }
    if required_tools.issubset(successful_tools) and all(
        step.get("success") is True for step in calls
    ):
        return 5
    if successful_tools:
        return 3
    return 1


def judge_run(result: dict, judge_model: str) -> dict:
    task_by_id = {task["id"]: task for task in TASKS}
    result_by_id = {item["task_id"]: item for item in result["results"]}

    payload = [
        {
            "task_id": item["task_id"],
            "category": item["category"],
            "prompt": task_by_id.get(item["task_id"], {}).get("prompt", ""),
            "reference_answer": item["expected"],
            "agent_answer": item["output"],
            "execution_trace": item["trace"],
            "status": item["status"],
        }
        for item in result["results"]
    ]

    prompt = f"""
You are an impartial evaluator for LLM agent benchmark results.
Evaluate every supplied task exactly once.
Return only JSON in this shape:
{{"evaluations":[{{"task_id":"id","correctness":1,"reasoning_quality":1,"overall":1,"rationale":"brief explanation"}}]}}
Scores must be JSON integers from 1 (poor) to 5 (excellent).
Correctness: does the answer satisfy the task and agree with the reference?
Reasoning quality: does the answer and trace support a sound conclusion?
Overall: your holistic score for the task.
Do not evaluate tool-use quality; it is calculated separately from the recorded execution trace.
Do not omit any task_id.
DATA:
{json.dumps(payload, ensure_ascii=False)}
"""

    response = httpx.post(
        f"{settings.ollama_base_url.rstrip('/')}/api/generate",
        json={
            "model": judge_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        },
        timeout=max(settings.request_timeout_seconds, 120),
    )
    response.raise_for_status()
    raw = response.json().get("response", "")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge returned invalid JSON: {raw[:500]}") from exc

    evaluations = parsed.get("evaluations") if isinstance(parsed, dict) else None
    if not isinstance(evaluations, list):
        raise ValueError(f"Judge response must contain an evaluations array: {raw[:500]}")

    expected_ids = set(result_by_id)
    normalized = {}

    for item in evaluations:
        if not isinstance(item, dict):
            continue
        task_id = item.get("task_id")
        if task_id not in expected_ids:
            continue

        scores = {}
        for key in ("correctness", "reasoning_quality", "overall"):
            value = item.get(key)
            if isinstance(value, str) and value.strip().isdigit():
                value = int(value.strip())
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
                raise ValueError(f"Judge returned invalid {key} score for {task_id}")
            scores[key] = value

        task = task_by_id.get(task_id, {})
        trace = result_by_id[task_id].get("trace", [])
        scores["tool_use_quality"] = _deterministic_tool_score(task, trace)

        normalized[task_id] = {
            "task_id": task_id,
            **scores,
            "rationale": str(item.get("rationale", ""))[:1000],
        }

    if set(normalized) != expected_ids:
        missing = sorted(expected_ids - set(normalized))
        raise ValueError(f"Judge omitted task evaluations: {', '.join(missing)}")

    average = sum(item["overall"] for item in normalized.values()) / len(normalized)
    return {
        "judge_model": judge_model,
        "average_overall_score": round(average, 2),
        "evaluations": [normalized[item["task_id"]] for item in result["results"]],
        "note": "Tool-use quality is calculated deterministically from each task's allowed tools and recorded execution trace. Correctness, reasoning quality, and overall scores are supplementary LLM-judge assessments.",
    }
