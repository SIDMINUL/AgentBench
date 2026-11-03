import json
import httpx

from app.config import settings
from app.tasks import TASKS


def judge_run(result: dict, judge_model: str) -> dict:
    task_prompts = {task["id"]: task["prompt"] for task in TASKS}
    payload = [
        {
            "task_id": item["task_id"],
            "category": item["category"],
            "prompt": task_prompts.get(item["task_id"], ""),
            "reference_answer": item["expected"],
            "agent_answer": item["output"],
            "execution_trace": item["trace"],
            "status": item["status"],
        }
        for item in result["results"]
    ]
    prompt = f"""
You are an impartial evaluator for LLM agent benchmark results.
Evaluate each item using only its task, reference answer, agent answer, and execution trace.
Return one JSON object in this exact shape:
{{"evaluations":[{{"task_id":"id","correctness":1,"reasoning_quality":1,"tool_use_quality":1,"overall":1,"rationale":"brief explanation"}}]}}
Scores must be integers from 1 (poor) to 5 (excellent).
Correctness: does the answer satisfy the task and agree with the reference?
Reasoning quality: does the answer and trace support a sound conclusion?
Tool-use quality: were tools, arguments, and observations used appropriately? For tasks requiring no tools, score 5 if none were needed.
Do not reward unsupported claims of tool execution. Score failed or empty answers low.
Include exactly one evaluation per supplied task and preserve task_id.
DATA:
{json.dumps(payload, ensure_ascii=False)}
"""
    response = httpx.post(
        f"{settings.ollama_base_url.rstrip('/')}/api/generate",
        json={"model": judge_model, "prompt": prompt, "stream": False,
              "format": "json", "options": {"temperature": 0}},
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

    expected_ids = {item["task_id"] for item in result["results"]}
    normalized = {}
    for item in evaluations:
        if not isinstance(item, dict) or item.get("task_id") not in expected_ids:
            continue
        scores = {}
        for key in ("correctness", "reasoning_quality", "tool_use_quality", "overall"):
            value = item.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
                raise ValueError(f"Judge returned invalid {key} score for {item.get('task_id')}")
            scores[key] = value
        normalized[item["task_id"]] = {
            "task_id": item["task_id"], **scores,
            "rationale": str(item.get("rationale", ""))[:1000],
        }
    if set(normalized) != expected_ids:
        missing = sorted(expected_ids - set(normalized))
        raise ValueError(f"Judge omitted task evaluations: {', '.join(missing)}")
    average = sum(item["overall"] for item in normalized.values()) / len(normalized) if normalized else 0
    return {
        "judge_model": judge_model,
        "average_overall_score": round(average, 2),
        "evaluations": [normalized[item["task_id"]] for item in result["results"]],
        "note": "LLM-judge scores are supplementary and do not replace deterministic benchmark metrics.",
    }
