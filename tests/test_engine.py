from app.engine import run, run_task
from app.storage import get_run, list_runs, save_run
from app.tasks import TASKS
from app.providers import MockProvider


def test_mock_benchmark_passes():
    result = run("mock", "mock-agent")
    assert result["total_tasks"] == 3
    assert result["task_success_rate"] == 1.0


def test_tool_task_records_execution_trace():
    task = next(task for task in TASKS if task["id"] == "multi_001")
    result = run_task(task, MockProvider(), "mock", "mock-agent")
    assert result["success"] is True
    assert result["tool_calls"] == 1
    assert result["tool_errors"] == 0
    assert result["trace"][0]["tool"] == "product_lookup"
    assert result["trace"][-1]["type"] == "final"


def test_task_cannot_call_disallowed_tool():
    class BadProvider:
        def next_action(self, task, history, model):
            return {"type": "tool_call", "tool": "dangerous_shell", "arguments": {}}

    task = next(task for task in TASKS if task["id"] == "multi_001")
    result = run_task(task, BadProvider(), "mock", "bad")
    assert result["success"] is False
    assert "Tool not allowed" in result["error"]


def test_benchmark_run_persistence(tmp_path, monkeypatch):
    from app import storage

    monkeypatch.setattr(storage, "DATABASE_PATH", tmp_path / "test_agentbench.db")
    original = run("mock", "mock-agent")
    saved = save_run(original)

    assert saved["run_id"] == 1
    assert saved["created_at"]
    assert get_run(saved["run_id"])["task_success_rate"] == 1.0
    history = list_runs()
    assert len(history) == 1
    assert history[0]["provider"] == "mock"


def test_compare_endpoint_runs_and_saves_each_model(monkeypatch, tmp_path):
    from app import main, storage
    from fastapi.testclient import TestClient

    monkeypatch.setattr(storage, "DATABASE_PATH", tmp_path / "comparison.db")
    monkeypatch.setattr(main, "run", lambda provider, model: {
        "provider": provider,
        "model": model,
        "task_success_rate": 1.0 if model == "model-a" else 0.5,
        "successful_tasks": 3 if model == "model-a" else 1,
        "total_tasks": 3,
        "average_latency_ms": 100.0 if model == "model-a" else 50.0,
        "p95_latency_ms": 120.0,
        "tool_calls": 1,
        "tool_errors": 0,
        "tool_call_success_rate": 1.0,
        "results": [],
    })

    response = TestClient(main.app).post(
        "/api/benchmark/compare",
        json={"provider": "ollama", "models": ["model-a", "model-b"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["comparison_count"] == 2
    assert data["ranking"][0]["model"] == "model-a"
    assert len(storage.list_runs()) == 2


def test_compare_endpoint_rejects_duplicate_models():
    from app.main import app
    from fastapi.testclient import TestClient

    response = TestClient(app).post(
        "/api/benchmark/compare",
        json={"provider": "ollama", "models": ["llama3.2:3b", "llama3.2:3b"]},
    )
    assert response.status_code == 400
