import json
import httpx

from app.config import settings
from app.tasks import TASKS


def _deterministic_tool_score(task: dict, trace: list) -> int:
    required_tools = set(task.get("tools", []))
    calls = [step for step in trace if step.get("type") == "tool_call"]

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
You are an impartial evaluator of reasoning quality for LLM agent benchmark results.
Evaluate every supplied task exactly once.
Return only JSON in this shape:
{{"evaluations":[{{"task_id":"id","reasoning_quality":1,"rationale":"brief explanation"}}]}}
reasoning_quality must be a JSON integer from 1 (poor) to 5 (excellent).
Assess whether the execution trace shows a sound, coherent process leading to the answer.
Do not score correctness, overall quality, or tool use; those are calculated independently.
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
    overall_score_sum = 0.0

    for item in evaluations:
        if not isinstance(item, dict):
            continue
        task_id = item.get("task_id")
        if task_id not in expected_ids:
            continue

        reasoning_quality = item.get("reasoning_quality")
        if isinstance(reasoning_quality, str) and reasoning_quality.strip().isdigit():
            reasoning_quality = int(reasoning_quality.strip())
        if (
            isinstance(reasoning_quality, bool)
            or not isinstance(reasoning_quality, int)
            or not 1 <= reasoning_quality <= 5
        ):
            raise ValueError(
                f"Judge returned invalid reasoning_quality score for {task_id}"
            )

        recorded = result_by_id[task_id]
        task = task_by_id.get(task_id, {})
        correctness = 5 if recorded.get("success") is True else 1
        tool_use_quality = _deterministic_tool_score(
            task,
            recorded.get("trace", []),
        )
        raw_overall = (correctness + reasoning_quality + tool_use_quality) / 3
        overall_score_sum += raw_overall
        overall = round(raw_overall, 2)

        normalized[task_id] = {
            "task_id": task_id,
            "correctness": correctness,
            "reasoning_quality": reasoning_quality,
            "overall": overall,
            "tool_use_quality": tool_use_quality,
            "rationale": str(item.get("rationale", ""))[:1000],
        }

    if set(normalized) != expected_ids:
        missing = sorted(expected_ids - set(normalized))
        raise ValueError(f"Judge omitted task evaluations: {', '.join(missing)}")

    # Average unrounded task scores to avoid cumulative rounding drift.
    average = overall_score_sum / len(normalized)
    return {
        "judge_model": judge_model,
        "average_overall_score": round(average, 2),
        "evaluations": [normalized[item["task_id"]] for item in result["results"]],
        "note": "Correctness is derived from the benchmark's deterministic success flag. Tool-use quality is calculated from allowed tools and the recorded execution trace. The LLM judge scores reasoning quality only; overall is the mean of these three dimensions.",
    }
