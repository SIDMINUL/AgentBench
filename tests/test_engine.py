from app.engine import run, run_task
from app.judge import judge_run
from app.storage import get_run, list_runs, save_run, update_run
from app.tasks import TASKS
from app.providers import MockProvider


def test_mock_benchmark_passes_all_tasks():
    result = run("mock", "mock-agent")
    assert result["total_tasks"] == len(TASKS) == 12
    assert result["task_success_rate"] == 1.0
    assert result["successful_tasks"] == 10


def test_benchmark_covers_all_task_categories():
    categories = {task["category"] for task in TASKS}
    assert categories == {
        "reasoning", "planning", "tool_use", "failure_recovery"
    }



def test_benchmark_reports_category_level_metrics():
    result = run("mock", "mock-agent")
    assert result["category_metrics"]["reasoning"] == {
        "total_tasks": 4, "successful_tasks": 4, "task_success_rate": 1.0
    }
    assert result["category_metrics"]["planning"]["total_tasks"] == 2
    assert result["category_metrics"]["tool_use"]["successful_tasks"] == 3
    assert result["category_metrics"]["failure_recovery"] == {
        "total_tasks": 3, "successful_tasks": 3, "task_success_rate": 1.0
    }


def test_ollama_recovery_prompt_requires_fallback_lookup(monkeypatch):
    import json
    from app.providers import OllamaProvider

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": json.dumps({
                "type": "tool_call",
                "tool": "product_lookup",
                "arguments": {"sku": "SKU-RED-01"},
            })}

    def fake_post(url, json=None, timeout=None):
        captured["payload"] = json
        return FakeResponse()

    monkeypatch.setattr("app.providers.httpx.post", fake_post)
    task = next(task for task in TASKS if task["id"] == "recovery_001")
    history = [{
        "type": "tool_error",
        "tool": "product_lookup",
        "error": "Unknown SKU: SKU-MISSING-99",
    }]
    action = OllamaProvider().next_action(task, history, "test-model")
    prompt = captured["payload"]["prompt"]

    assert action == {
        "type": "tool_call",
        "tool": "product_lookup",
        "arguments": {"sku": "SKU-RED-01"},
    }
    assert "does not tell you the product's stock" in prompt
    assert "you MUST make that tool_call" in prompt
    assert "Do not return a final answer while a required recovery lookup is still outstanding" in prompt


def test_ollama_reasoning_prompt_includes_verification_guidance(monkeypatch):
    import json
    from app.providers import OllamaProvider

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": json.dumps({"type": "final", "answer": "25"})}

    def fake_post(url, json=None, timeout=None):
        captured["payload"] = json
        return FakeResponse()

    monkeypatch.setattr("app.providers.httpx.post", fake_post)
    task = next(task for task in TASKS if task["id"] == "math_002")
    action = OllamaProvider().next_action(task, [], "test-model")
    prompt = captured["payload"]["prompt"]

    assert action == {"type": "final", "answer": "25"}
    assert "fraction of the original quantity" in prompt
    assert "recomputing it using a second simple method" in prompt
    assert "No tools are available" in prompt


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
    assert evaluation["average_overall_score"] == 4.28


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
    assert result["tool_calls"] == 3
    assert result["duplicate_tool_calls"] == 2
    assert len(executions) == 1
    assert result["trace"][-1]["type"] == "tool_call_duplicate"
    assert result["status"] == "repeated_action"
    assert "twice" in result["error"]


def test_recovery_task_still_allows_different_tool_arguments():
    task = next(task for task in TASKS if task["id"] == "recovery_001")
    result = run_task(task, MockProvider(), "mock", "mock-agent")

    assert result["success"] is True
    assert result["tool_calls"] == 2
    assert result["duplicate_tool_calls"] == 0



def test_agent_can_use_duplicate_call_feedback_to_finish():
    task = next(task for task in TASKS if task["id"] == "multi_001")

    class FeedbackAwareProvider:
        def next_action(self, task, history, model):
            if any(item.get("type") == "duplicate_tool_call" for item in history):
                return {"type": "final", "answer": "REORDER"}
            return {
                "type": "tool_call",
                "tool": "product_lookup",
                "arguments": {"sku": "SKU-BLUE-02"},
            }

    result = run_task(task, FeedbackAwareProvider(), "mock", "feedback-aware")

    assert result["success"] is True
    assert result["tool_calls"] == 2
    assert result["duplicate_tool_calls"] == 1
    assert result["trace"][-1]["type"] == "final"



def test_repeated_failed_tool_call_is_suppressed():
    task = next(task for task in TASKS if task["id"] == "recovery_001")

    class RepeatingFailureProvider:
        def next_action(self, task, history, model):
            return {
                "type": "tool_call",
                "tool": "product_lookup",
                "arguments": {"sku": "SKU-MISSING-99"},
            }

    result = run_task(task, RepeatingFailureProvider(), "mock", "failure-repeater")

    assert result["success"] is False
    assert result["status"] == "repeated_action"
    assert result["tool_calls"] == 3
    assert result["tool_errors"] == 1
    assert result["duplicate_tool_calls"] == 2
    assert sum(step["type"] == "tool_call" for step in result["trace"]) == 1
    assert result["trace"][-1]["type"] == "tool_call_duplicate"

def test_duplicate_call_replays_cached_result_as_normal_observation():
    task = next(task for task in TASKS if task["id"] == "multi_001")
    observed_histories = []

    class ReplayAwareProvider:
        def next_action(self, task, history, model):
            observed_histories.append(list(history))
            duplicate = any(item.get("type") == "duplicate_tool_call" for item in history)
            if duplicate:
                replayed = [
                    item for item in history
                    if item.get("type") == "tool_result" and item.get("replayed")
                ]
                assert replayed
                assert replayed[-1]["result"]["stock"] == 4
                return {"type": "final", "answer": "REORDER"}
            return {
                "type": "tool_call",
                "tool": "product_lookup",
                "arguments": {"sku": "SKU-BLUE-02"},
            }

    result = run_task(task, ReplayAwareProvider(), "mock", "replay-aware")
    assert result["success"] is True
    assert result["duplicate_tool_calls"] == 1

def test_all_failure_recovery_fixtures_pass_with_mock_provider():
    recovery_tasks = [task for task in TASKS if task["category"] == "failure_recovery"]
    assert len(recovery_tasks) == 3
    for task in recovery_tasks:
        result = run_task(task, MockProvider(), "mock", "mock-agent")
        assert result["success"] is True
        assert result["tool_errors"] == 1
        assert result["tool_calls"] == 2
        assert result["trace"][-1]["type"] == "final"


def test_engine_blocks_final_until_required_recovery_succeeds():
    task = next(task for task in TASKS if task["id"] == "recovery_001")

    class PrematureFinalProvider:
        def next_action(self, task, history, model):
            if not history:
                return {
                    "type": "tool_call",
                    "tool": "product_lookup",
                    "arguments": {"sku": "SKU-MISSING-99"},
                }
            if any(item.get("type") == "recovery_required" for item in history):
                if not any(
                    item.get("type") == "tool_result"
                    and item.get("result", {}).get("sku") == "SKU-RED-01"
                    for item in history
                ):
                    return {
                        "type": "tool_call",
                        "tool": "product_lookup",
                        "arguments": {"sku": "SKU-RED-01"},
                    }
                return {"type": "final", "answer": "AVAILABLE"}
            return {"type": "final", "answer": "OUT_OF_STOCK"}

    result = run_task(task, PrematureFinalProvider(), "mock", "guard-test")
    assert result["success"] is True
    assert any(step["type"] == "final_blocked_recovery" for step in result["trace"])
    assert result["trace"][-1]["type"] == "final"