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
        instructions = {
            "task": task["prompt"],
            "allowed_tools": {
                "product_lookup": {
                    "description": "Look up an inventory item by SKU.",
                    "arguments": {"sku": "string"},
                }
            },
            "history": history,
            "required_json": {
                "tool_call": {"type": "tool_call", "tool": "product_lookup", "arguments": {"sku": "SKU-RED-01"}},
                "final": {"type": "final", "answer": "your answer"},
            },
            "instruction": "Choose one next action. Return only valid JSON. Never invent tool results. Use only listed tools.",
        }
        response = httpx.post(
            f"{settings.ollama_base_url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": json.dumps(instructions), "stream": False, "format": "json"},
            timeout=settings.request_timeout_seconds,
        )
        response.raise_for_status()
        raw = response.json().get("response", "")
        action = json.loads(raw)
        if not isinstance(action, dict) or action.get("type") not in {"tool_call", "final"}:
            raise ValueError("Model returned an invalid action schema")
        return action

def get_provider(name: str):
    if name == "mock":
        return MockProvider()
    if name == "ollama":
        return OllamaProvider()
    raise ValueError(f"Unsupported provider: {name}")
