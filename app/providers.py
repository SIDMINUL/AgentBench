import json
import httpx
from app.config import settings

class MockProvider:
    """Deterministic agent used to test orchestration, not model quality."""
    def next_action(self, task, history, model):
        if task["id"] == "multi_001":
            if not any(item.get("type") == "tool_result" for item in history):
                return {
                    "type": "tool_call",
                    "tool": "product_lookup",
                    "arguments": {"sku": "SKU-BLUE-02"},
                }
            result = next(item["result"] for item in reversed(history) if item.get("type") == "tool_result")
            return {"type": "final", "answer": "REORDER" if result["stock"] < 10 else "STOCK_OK"}
        if task["id"] == "math_001":
            return {"type": "final", "answer": "15"}
        if task["id"] == "logic_001":
            return {"type": "final", "answer": "10"}
        return {"type": "final", "answer": ""}


class OllamaProvider:
    def next_action(self, task, history, model):
        allowed_tools = (
            "product_lookup(sku: string) — returns product name and stock count."
            if "product_lookup" in task.get("tools", [])
            else "No tools are available for this task."
        )
        prompt = f"""
You are an agent completing one benchmark task.
TASK: {task["prompt"]}
ALLOWED TOOLS: {allowed_tools}
OBSERVATIONS SO FAR: {json.dumps(history)}

Choose exactly one action. Return one JSON object and no other text.
To call a tool, use:
{{"type":"tool_call","tool":"product_lookup","arguments":{{"sku":"SKU-BLUE-02"}}}}
To finish, use:
{{"type":"final","answer":"your concise answer"}}
Do not claim you called a tool unless you return a tool_call action.
Do not invent tool results. Only use tools listed above.
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
            # Plain text is still a usable final response for non-tool tasks.
            if "product_lookup" not in task.get("tools", []):
                return {"type": "final", "answer": raw}
            raise ValueError(f"Model did not return valid JSON: {raw[:500]}")

        if not isinstance(action, dict):
            raise ValueError(f"Model action must be a JSON object: {raw[:500]}")

        # Normalize common model variations into AgentBench's canonical schema.
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

        # Some models wrap the selected action in a nested field.
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
