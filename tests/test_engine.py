from app.engine import run, run_task
from app.judge import judge_run
from app.storage import get_run, list_runs, save_run, update_run
from app.tasks import TASKS
from app.providers import MockProvider


def test_mock_benchmark_passes_all_tasks():
    result = run("mock", "mock-agent")
    assert result["total_tasks"] == len(TASKS) == 10
    assert result["task_success_rate"] == 1.0
    assert result["successful_tasks"] == 10


def test_benchmark_covers_all_task_categories():
    categories = {task["category"] for task in TASKS}
    assert categories == {
        "reasoning", "planning", "tool_use", "failure_recovery"
    }


def test_tool_task_records_execution_trace():
    task = next(task for task in TASKS if task["id"] == "multi_001")
    result = run_task(task, MockProvider(), "mock", "mock-agent")
    assert result["success"] is True
    assert result["tool_calls"] == 1
    assert result["tool_errors"] == 0
    assert result["trace"][0]["tool"] == "product_lookup"
    assert result["trace"][-1]["type"] == "final"


def test_multi_tool_task_uses_both_products():
    task = next(task for task in TASKS if task["id"] == "multi_003")
    result = run_task(task, MockProvider(), "mock", "mock-agent")
    successful_lookups = [
        step for step in result["trace"]
        if step["type"] == "tool_call" and step["success"]
    ]
    assert result["success"] is True
    assert result["output"] == "SKU-RED-01"
    assert len(successful_lookups) == 2


def test_failure_recovery_records_error_then_recovers():
    task = next(task for task in TASKS if task["id"] == "recovery_001")
    result = run_task(task, MockProvider(), "mock", "mock-agent")
    assert result["success"] is True
    assert result["tool_errors"] == 1
    assert result["tool_calls"] == 2
    assert result["trace"][0]["success"] is False
    assert result["trace"][1]["success"] is True
    assert result["trace"][-1]["answer"] == "AVAILABLE"


def test_task_cannot_call_disallowed_tool():
    class BadProvider:
        def next_action(self, task, history, model):
            return {
                "type": "tool_call",
                "tool": "dangerous_shell",
                "arguments": {},
            }

    task = next(task for task in TASKS if task["id"] == "multi_001")
    result = run_task(task, BadProvider(), "mock", "bad")
    assert result["success"] is False
    assert "Tool not allowed" in result["error"]


def test_benchmark_run_persistence(tmp_path, monkeypatch):
    from app import storage
    monkeypatch.setattr(storage, "DATABASE_PATH", tmp_path / "test_agentbench.db")
    saved = save_run(run("mock", "mock-agent"))
    assert saved["run_id"] == 1
    assert saved["created_at"]
    assert get_run(saved["run_id"])["task_success_rate"] == 1.0
    assert len(list_runs()) == 1
    saved["llm_judge"] = {"average_overall_score": 4.0}
    assert update_run(saved["run_id"], saved)
    assert get_run(saved["run_id"])["llm_judge"]["average_overall_score"] == 4.0


def test_compare_endpoint_runs_and_saves_each_model(monkeypatch, tmp_path):
    from app import main, storage
    from fastapi.testclient import TestClient
    monkeypatch.setattr(storage, "DATABASE_PATH", tmp_path / "comparison.db")
    monkeypatch.setattr(main, "run", lambda provider, model: {
        "provider": provider, "model": model,
        "task_success_rate": 1.0 if model == "model-a" else 0.5,
        "successful_tasks": 10 if model == "model-a" else 5,
        "total_tasks": 10,
        "average_latency_ms": 100.0 if model == "model-a" else 50.0,
        "p95_latency_ms": 120.0, "tool_calls": 1, "tool_errors": 0,
        "tool_call_success_rate": 1.0, "results": []})
    response = TestClient(main.app).post(
        "/api/benchmark/compare",
        json={"provider": "ollama", "models": ["model-a", "model-b"]},
    )
    assert response.status_code == 200
    assert response.json()["ranking"][0]["model"] == "model-a"
    assert len(storage.list_runs()) == 2


def test_compare_endpoint_rejects_duplicate_models():
    from app.main import app
    from fastapi.testclient import TestClient
    response = TestClient(app).post(
        "/api/benchmark/compare",
        json={"provider": "ollama", "models": ["llama3.2:3b", "llama3.2:3b"]},
    )
    assert response.status_code == 400


def test_judge_run_uses_deterministic_correctness_and_tool_scores(monkeypatch):
    from app import judge

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            items = [
                {
                    "task_id": task["id"],
                    "reasoning_quality": 3,
                    "rationale": "Reasonable.",
                }
                for task in TASKS
            ]
            import json
            return {"response": json.dumps({"evaluations": items})}

    monkeypatch.setattr(
        judge.httpx, "post", lambda *args, **kwargs: FakeResponse()
    )
    evaluation = judge_run(run("mock", "mock-agent"), "judge-model")
    assert len(evaluation["evaluations"]) == len(TASKS)
    assert evaluation["evaluations"][0]["correctness"] == 5
    recovery = next(
        item for item in evaluation["evaluations"]
        if item["task_id"] == "recovery_001"
    )
    assert recovery["tool_use_quality"] == 3
    assert evaluation["average_overall_score"] == 4.27


def test_repeated_successful_tool_call_is_suppressed(monkeypatch):
    import app.engine as engine

    task = next(task for task in TASKS if task["id"] == "multi_001")
    executions = []
    original_execute = engine.execute_tool

    def counting_execute(tool_name, arguments):
        executions.append((tool_name, arguments.copy()))
        return original_execute(tool_name, arguments)

    monkeypatch.setattr(engine, "execute_tool", counting_execute)

    class RepeatingProvider:
        def next_action(self, task, history, model):
            return {
                "type": "tool_call",
                "tool": "product_lookup",
                "arguments": {"sku": "SKU-BLUE-02"},
            }

    result = run_task(task, RepeatingProvider(), "mock", "repeater")

    assert result["success"] is False
    assert result["status"] == "repeated_action"
    assert result["tool_calls"] == 2
    assert result["duplicate_tool_calls"] == 1
    assert len(executions) == 1
    assert result["trace"][-1]["type"] == "tool_call_duplicate"


def test_recovery_task_still_allows_different_tool_arguments():
    task = next(task for task in TASKS if task["id"] == "recovery_001")
    result = run_task(task, MockProvider(), "mock", "mock-agent")

    assert result["success"] is True
    assert result["tool_calls"] == 2
    assert result["duplicate_tool_calls"] == 0
