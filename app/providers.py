import json
import httpx
from app.config import settings


class MockProvider:
    """Deterministic agent for validating benchmark orchestration, not model quality."""

    def next_action(self, task, history, model):
        task_id = task["id"]

        static_answers = {
            "math_001": "15",
            "logic_001": "10",
            "math_002": "25",
            "logic_002": "Cara",
            "planning_001": "implementation, testing, deployment",
            "planning_002": "10:10",
        }
        if task_id in static_answers:
            return {"type": "final", "answer": static_answers[task_id]}

        tool_results = [
            item for item in history
            if item.get("type") == "tool_result"
            and item.get("tool") == "product_lookup"
        ]
        tool_errors = [
            item for item in history
            if item.get("type") == "tool_error"
            and item.get("tool") == "product_lookup"
        ]
        looked_up = {
            item["result"]["sku"]
            for item in tool_results
        }

        if task_id == "multi_001":
            if "SKU-BLUE-02" not in looked_up:
                return {
                    "type": "tool_call",
                    "tool": "product_lookup",
                    "arguments": {"sku": "SKU-BLUE-02"},
                }
            return {"type": "final", "answer": "REORDER"}

        if task_id == "multi_002":
            if "SKU-RED-01" not in looked_up:
                return {
                    "type": "tool_call",
                    "tool": "product_lookup",
                    "arguments": {"sku": "SKU-RED-01"},
                }
            return {"type": "final", "answer": "STOCK_OK"}

        if task_id == "multi_003":
            for sku in ("SKU-BLUE-02", "SKU-RED-01"):
                if sku not in looked_up:
                    return {
                        "type": "tool_call",
                        "tool": "product_lookup",
                        "arguments": {"sku": sku},
                    }
            stock_by_sku = {
                item["result"]["sku"]: item["result"]["stock"]
                for item in tool_results
            }
            winner = max(stock_by_sku, key=stock_by_sku.get)
            return {"type": "final", "answer": winner}

        if task_id == "recovery_001":
            if not tool_errors and not looked_up:
                return {
                    "type": "tool_call",
                    "tool": "product_lookup",
                    "arguments": {"sku": "SKU-MISSING-99"},
                }
            if "SKU-RED-01" not in looked_up:
                return {
                    "type": "tool_call",
                    "tool": "product_lookup",
                    "arguments": {"sku": "SKU-RED-01"},
                }
            recovered = next(
                item["result"] for item in reversed(tool_results)
                if item["result"]["sku"] == "SKU-RED-01"
            )
            return {
                "type": "final",
                "answer": "AVAILABLE" if recovered["stock"] > 0 else "OUT_OF_STOCK",
            }

        return {"type": "final", "answer": ""}


class OllamaProvider:
    def next_action(self, task, history, model):
        allowed_tools = (
            "product_lookup(sku: string) — returns product name and stock count."
            if "product_lookup" in task.get("tools", [])
            else "No tools are available for this task."
        )
        tool_observations = [
            item for item in history
            if item.get("type") in {"tool_result", "tool_error"}
        ]
        reasoning_guidance = ""
        if not task.get("tools"):
            reasoning_guidance = """
REASONING AND VERIFICATION RULES:
- Solve the task carefully before producing the final answer. Do not guess or rush.
- For arithmetic, identify the starting quantity and apply each operation in the order stated.
- For fractions, calculate the fraction of the original quantity before applying later changes.
- For elapsed-time problems, add each duration sequentially to the start time; convert minutes past 60 into the next hour.
- For relational logic, translate each comparison into an ordering before selecting the answer.
- Check the result once by recomputing it using a second simple method.
- Return only the exact requested answer format. Do not include workings unless asked.
"""
        prompt = f"""
You are a reliable agent completing one benchmark task.
TASK: {task["prompt"]}
ALLOWED TOOLS: {allowed_tools}
TOOL OBSERVATIONS (authoritative; use these instead of repeating calls): {json.dumps(tool_observations)}
FULL ACTION HISTORY: {json.dumps(history)}
{reasoning_guidance}

STRICT ACTION RULES:
- If no tools are available, return a final action. Never call any tool.
- Only call a tool explicitly listed in ALLOWED TOOLS. Never invent tools.
- A tool_result already contains the answer from that tool. Read its result fields and reason from them.
- If a tool observation answers the task, return a final action now. Do not call the tool again.
- If an observation has "replayed": true, it is the cached result of an earlier call. Treat it exactly like a normal tool_result or tool_error, and do not retry it.
- Never repeat a tool call with identical arguments, even if the previous call was suppressed.
- If a tool returns an error, follow the task's recovery instruction using different arguments. Do not retry the same failed arguments.
- When the observations contain all information needed, return a final answer immediately.
- Return exactly one of these schemas:
{{"type":"tool_call","tool":"product_lookup","arguments":{{"sku":"SKU-BLUE-02"}}}}
{{"type":"final","answer":"your concise answer"}}
Do not invent tool results or claim a tool was called unless you return a tool_call action.
"""
        response = httpx.post(
            f"{settings.ollama_base_url.rstrip('/')}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            },
            timeout=settings.request_timeout_seconds,
        )
        response.raise_for_status()
        raw = response.json().get("response", "").strip()

        try:
            action = json.loads(raw)
        except json.JSONDecodeError:
            if not task.get("tools"):
                return {"type": "final", "answer": raw}
            raise ValueError(f"Model did not return valid JSON: {raw[:500]}")

        if not isinstance(action, dict):
            raise ValueError(f"Model action must be a JSON object: {raw[:500]}")

        action_type = action.get("type") or action.get("action")
        if action_type in {"final_answer", "answer", "respond"}:
            action_type = "final"
        if action_type in {"tool", "call_tool", "function_call"}:
            action_type = "tool_call"

        if action_type == "final":
            answer = action.get("answer", action.get("content", action.get("final_answer", "")))
            return {"type": "final", "answer": str(answer)}

        if action_type == "tool_call":
            tool_name = action.get("tool", action.get("tool_name", action.get("name")))
            arguments = action.get("arguments", action.get("tool_input", action.get("parameters", {})))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    raise ValueError(f"Tool arguments were not valid JSON: {arguments[:300]}")
            if not isinstance(arguments, dict):
                raise ValueError(f"Tool arguments must be a JSON object: {raw[:500]}")
            return {"type": "tool_call", "tool": tool_name, "arguments": arguments}

        nested = action.get("tool_call")
        if isinstance(nested, dict):
            return {
                "type": "tool_call",
                "tool": nested.get("tool", nested.get("name")),
                "arguments": nested.get("arguments", nested.get("parameters", {})),
            }

        raise ValueError(f"Model returned an unsupported action schema: {raw[:500]}")


def get_provider(name: str):
    if name == "mock":
        return MockProvider()
    if name == "ollama":
        return OllamaProvider()
    raise ValueError(f"Unsupported provider: {name}")
