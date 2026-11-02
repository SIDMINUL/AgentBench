import httpx
from app.config import settings

class MockProvider:
    """Deterministic provider for pipeline tests, not model-quality measurement."""
    def answer(self, prompt: str, model: str) -> str:
        if "notebooks" in prompt:
            return "15"
        if "pens" in prompt:
            return "10"
        if "reorder threshold" in prompt:
            return "REORDER"
        return ""

class OllamaProvider:
    def answer(self, prompt: str, model: str) -> str:
        response = httpx.post(
            f"{settings.ollama_base_url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=settings.request_timeout_seconds,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

def get_provider(name: str):
    if name == "mock":
        return MockProvider()
    if name == "ollama":
        return OllamaProvider()
    raise ValueError(f"Unsupported provider: {name}")
