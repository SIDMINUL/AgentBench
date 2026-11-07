from app.engine import run, run_task
from app.judge import judge_run
from app.storage import get_run, list_runs, save_run, update_run
from app.tasks import TASKS
from app.providers import MockProvider


def test_mock_benchmark_passes_all_tasks():
    result = run("mock", "mock-agent")
    assert result["total_tasks"] == len(TASKS) == 22
    assert result["task_success_rate"] == 1.0
    assert result["successful_tasks"] == 22


def test_benchmark_covers_all_task_categories():
    categories = {task["category"] for task in TASKS}
    assert categories == {
        "reasoning", "planning", "tool_use", "failure_recovery"
    }



def test_benchmark_reports_category_level_metrics():
    result = run("mock", "mock-agent")
    assert result["category_metrics"]["reasoning"] == {
        "total_tasks": 8,
        "attempted_tasks": 8,
        "skipped_tasks": 0,
        "successful_tasks": 8,
        "task_success_rate": 1.0,
    }
    assert result["category_metrics"]["planning"]["total_tasks"] == 6
    assert result["category_metrics"]["tool_use"]["successful_tasks"] == 5
    assert result["category_metrics"]["failure_recovery"] == {
        "total_tasks": 3,
        "attempted_tasks": 3,
        "skipped_tasks": 0,
        "successful_tasks": 3,
        "task_success_rate": 1.0,
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
        captured.setdefault("payload", json)
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
        captured.setdefault("payload", json)
        return FakeResponse()

    monkeypatch.setattr("app.providers.httpx.post", fake_post)
    task = next(task for task in TASKS if task["id"] == "math_002")
    action = OllamaProvider().next_action(task, [], "test-model")
    prompt = captured["payload"]["prompt"]

    assert action == {"type": "final", "answer": "25"}
    assert "fraction changes" in prompt
    assert "independently check the result" in prompt
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
        "successful_tasks": 22 if model == "model-a" else 11,
        "total_tasks": 22,
        "average_latency_ms": 100.0 if model == "model-a" else 50.0,
        "p95_latency_ms": 120.0, "tool_calls": 1, "tool_errors": 0,
        "tool_call_success_rate": 1.0, "results": []})
    response = TestClient(main.app).post(
        "/api/benchmark/compare",
        json={
            "provider": "ollama",
            "models": ["model-a", "model-b"],
            "repetitions": 1,
        },
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
    from app.providers import settings

    monkeypatch.setattr(settings, "groq_api_key", "test-groq-key")

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
            return {
                "choices": [{
                    "message": {
                        "content": json.dumps({"evaluations": items})
                    }
                }]
            }

    monkeypatch.setattr(
        "app.providers.httpx.post", lambda *args, **kwargs: FakeResponse()
    )
    evaluation = judge_run(run("mock", "mock-agent"), "judge-model")
    assert len(evaluation["evaluations"]) == len(TASKS)
    assert evaluation["evaluations"][0]["correctness"] == 5
    recovery = next(
        item for item in evaluation["evaluations"]
        if item["task_id"] == "recovery_001"
    )
    assert recovery["tool_use_quality"] == 3
    assert evaluation["average_overall_score"] == 4.24


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


def test_engine_blocks_multi_tool_final_until_all_required_lookups_succeed():
    task = next(task for task in TASKS if task["id"] == "multi_003")

    class EarlyFinalProvider:
        def next_action(self, task, history, model):
            completed = {
                item["result"]["sku"]
                for item in history if item.get("type") == "tool_result"
            }
            if not any(item.get("type") == "required_tool_calls" for item in history):
                return {"type": "final", "answer": "SKU-BLUE-02"}
            for sku in ("SKU-BLUE-02", "SKU-RED-01"):
                if sku not in completed:
                    return {
                        "type": "tool_call",
                        "tool": "product_lookup",
                        "arguments": {"sku": sku},
                    }
            return {"type": "final", "answer": "SKU-RED-01"}

    result = run_task(task, EarlyFinalProvider(), "mock", "multi-tool-guard")
    assert result["success"] is True
    assert result["tool_calls"] == 2
    assert any(
        item["type"] == "final_blocked_required_tools"
        for item in result["trace"]
    )
    assert result["trace"][-1]["type"] == "final"



def test_ollama_tool_prompt_explains_numeric_thresholds(monkeypatch):
    import json
    from app.providers import OllamaProvider

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return { "response": json.dumps({"type": "final", "answer": "REORDER"}) }

    def fake_post(url, json=None, timeout=None):
        captured.setdefault("payload", json)
        return FakeResponse()

    monkeypatch.setattr("app.providers.httpx.post", fake_post)
    task = next(task for task in TASKS if task["id"] == "recovery_002")
    history = [
        {"type": "tool_error", "tool": "product_lookup", "error": "Unknown SKU"},
        {
            "type": "tool_result",
            "tool": "product_lookup",
            "result": {"sku": "SKU-BLUE-02", "name": "Blue Notebook", "stock": 4},
        },
    ]

    OllamaProvider().next_action(task, history, "test-model")
    prompt = captured["payload"]["prompt"]

    assert '"below N" means value < N' in prompt
    assert '"at least N" means value >= N' in prompt
    assert "a stock value of 4 is below 10" in prompt
    assert "Do not copy an answer pattern from another task" in prompt


def test_ollama_prompt_forces_final_after_observations_complete(monkeypatch):
    import json
    from app.providers import OllamaProvider

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": json.dumps({"type": "final", "answer": "REORDER"})}

    def fake_post(url, json=None, timeout=None):
        captured.setdefault("payload", json)
        return FakeResponse()

    monkeypatch.setattr("app.providers.httpx.post", fake_post)
    task = next(task for task in TASKS if task["id"] == "recovery_002")
    history = [
        {"type": "tool_error", "tool": "product_lookup", "error": "Unknown SKU"},
        {
            "type": "tool_result",
            "tool": "product_lookup",
            "result": {"sku": "SKU-BLUE-02", "name": "Blue Notebook", "stock": 4},
        },
        {
            "type": "tool_call_blocked_observations_complete",
            "message": "All required tool observations are already available.",
        },
    ]
    action = OllamaProvider().next_action(task, history, "test-model")
    prompt = captured["payload"]["prompt"]

    assert action == {"type": "final", "answer": "REORDER"}
    assert "FINAL ANSWER REQUIRED" in prompt
    assert "Your next action MUST be a final action" in prompt
    assert "use the successful fallback lookup result" in prompt


def test_ollama_prompt_lists_outstanding_required_tool_calls(monkeypatch):
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
                "arguments": {"sku": "SKU-BLUE-02"},
            })}

    def fake_post(url, json=None, timeout=None):
        captured.setdefault("payload", json)
        return FakeResponse()

    monkeypatch.setattr("app.providers.httpx.post", fake_post)
    task = next(task for task in TASKS if task["id"] == "multi_003")
    OllamaProvider().next_action(task, [], "test-model")
    prompt = captured["payload"]["prompt"]

    assert "SKU-BLUE-02" in prompt
    assert "SKU-RED-01" in prompt
    assert "Complete every listed required tool call" in prompt
    assert "Do not guess missing values" in prompt



def test_ollama_reasoning_prompt_has_explicit_count_fraction_and_time_checks(monkeypatch):
    import json
    from app.providers import OllamaProvider

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": json.dumps({"type": "final", "answer": "10"})}

    def fake_post(url, json=None, timeout=None):
        captured.setdefault("payload", json)
        return FakeResponse()

    monkeypatch.setattr("app.providers.httpx.post", fake_post)
    task = next(task for task in TASKS if task["id"] == "logic_001")
    OllamaProvider().next_action(task, [], "test-model")
    prompt = captured["payload"]["prompt"]

    assert "add every group's quantity first" in prompt
    assert "subtract only the items explicitly removed" in prompt
    assert "numeric expression internally" in prompt

    task = next(task for task in TASKS if task["id"] == "math_002")
    OllamaProvider().next_action(task, [], "test-model")
    prompt = captured["payload"]["prompt"]
    assert "unit fraction" in prompt
    assert "subtract borrowed/removed items, then apply returns" in prompt

    task = next(task for task in TASKS if task["id"] == "planning_002")
    OllamaProvider().next_action(task, [], "test-model")
    prompt = captured["payload"]["prompt"]
    assert "minutes after midnight" in prompt
    assert "including breaks" in prompt



def test_ollama_multi_tool_prompt_explains_numeric_comparison(monkeypatch):
    import json
    from app.providers import OllamaProvider

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": json.dumps({"type": "final", "answer": "SKU-RED-01"})}

    def fake_post(url, json=None, timeout=None):
        captured.setdefault("payload", json)
        return FakeResponse()

    monkeypatch.setattr("app.providers.httpx.post", fake_post)
    task = next(task for task in TASKS if task["id"] == "multi_003")
    history = [
        {"type": "tool_result", "tool": "product_lookup",
         "result": {"sku": "SKU-BLUE-02", "stock": 4}},
        {"type": "tool_result", "tool": "product_lookup",
         "result": {"sku": "SKU-RED-01", "stock": 27}},
    ]

    action = OllamaProvider().next_action(task, history, "test-model")
    prompt = captured["payload"]["prompt"]

    assert action == {"type": "final", "answer": "SKU-RED-01"}
    assert "Read the numeric stock field from EVERY required" in prompt
    assert "Compare the returned integers directly" in prompt
    assert "27 > 4" not in prompt
    assert "stock 4" not in prompt





def test_engine_retries_one_incorrect_final_answer():
    from app.engine import run_task

    class CorrectingProvider:
        def __init__(self):
            self.calls = 0

        def next_action(self, task, history, model):
            self.calls += 1
            if self.calls == 1:
                return {"type": "final", "answer": "11"}
            assert any(item.get("type") == "answer_retry_feedback" for item in history)
            return {"type": "final", "answer": "10"}

    task = next(task for task in TASKS if task["id"] == "logic_001")
    provider = CorrectingProvider()
    result = run_task(task, provider, "mock", "self-correction-test")

    assert result["success"] is True
    assert result["output"] == "10"
    assert result["answer_retry_used"] is True
    assert any(
        item["type"] == "final_answer_rejected_for_retry"
        for item in result["trace"]
    )


def test_independent_verifier_reports_candidate_and_suggested_correction():
    from app.engine import run_task

    class WrongAnswerProvider:
        def next_action(self, task, history, model):
            return {"type": "final", "answer": "7"}

    task = next(task for task in TASKS if task["id"] == "logic_001")
    result = run_task(task, WrongAnswerProvider(), "mock", "wrong-answer")
    assert result["output"] == "7"
    assert result["raw_output"] == "7"
    assert result["success"] is False
    assert result["verification"]["reference_output"] == "10"
    assert result["verification"]["candidate_correct"] is False
    assert result["verification"]["correction_suggested"] == "10"
    assert result["verification"]["correction_applied"] is False


def test_independent_verifier_recalculates_fraction_and_time():
    from app.engine import _independent_verification

    fraction_task = next(task for task in TASKS if task["id"] == "math_002")
    time_task = next(task for task in TASKS if task["id"] == "planning_002")
    fraction = _independent_verification(fraction_task, "21")
    elapsed = _independent_verification(time_task, "10:20")
    assert fraction["reference_output"] == "25"
    assert fraction["correction_suggested"] == "25"
    assert elapsed["reference_output"] == "10:10"
    assert elapsed["correction_suggested"] == "10:10"


def test_run_reports_independent_verification_coverage():
    result = run("mock", "mock-agent")
    assert result["verification_metrics"]["supported_tasks"] == 7
    assert result["verification_metrics"]["raw_correct"] == 7
    assert result["verification_metrics"]["raw_accuracy"] == 1.0
    assert result["verification_metrics"]["correction_applied"] is False



def test_compare_endpoint_aggregates_repeated_runs(monkeypatch, tmp_path):
    from app import main, storage
    from fastapi.testclient import TestClient

    monkeypatch.setattr(storage, "DATABASE_PATH", tmp_path / "repeated.db")
    counts = {"model-a": 0, "model-b": 0}

    def fake_run(provider, model):
        counts[model] += 1
        rate = (0.5, 1.0)[counts[model] - 1] if model == "model-a" else 0.75
        return {
            "provider": provider, "model": model,
            "task_success_rate": rate,
            "successful_tasks": int(rate * 22), "total_tasks": 22,
            "average_latency_ms": 100.0 if model == "model-a" else 50.0,
            "p95_latency_ms": 120.0, "tool_calls": 2, "tool_errors": 0,
            "tool_call_success_rate": 1.0, "category_metrics": {}, "results": [],
        }

    monkeypatch.setattr(main, "run", fake_run)
    response = TestClient(main.app).post(
        "/api/benchmark/compare",
        json={"provider": "ollama", "models": ["model-a", "model-b"], "repetitions": 2},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total_runs"] == 4
    assert data["repetitions"] == 2
    assert len(storage.list_runs()) == 4
    result_a = next(item for item in data["results"] if item["model"] == "model-a")
    assert result_a["mean_task_success_rate"] == 0.75
    assert result_a["success_rate_min"] == 0.5
    assert result_a["success_rate_max"] == 1.0
    assert len(result_a["run_ids"]) == 2
    assert data["ranking"][0]["model"] == "model-b"

def test_run_separates_model_independent_and_engine_assisted_successes():
    result = run("mock", "mock-agent")
    assert result["successful_tasks"] == 22
    assert result["model_independent_successful_tasks"] == 22
    assert result["engine_assisted_successful_tasks"] == 0
    assert result["model_independent_task_success_rate"] == 1.0
    multi = next(item for item in result["results"] if item["task_id"] == "multi_003")
    assert multi["model_initiated_tool_calls"] == 2
    assert multi["engine_assisted_completion"] is False
    assert multi["model_independent_success"] is True


def test_engine_assisted_completion_is_not_counted_as_model_independent():
    task = next(task for task in TASKS if task["id"] == "multi_003")

    class EarlyFinalProvider:
        def next_action(self, task, history, model):
            return {"type": "final", "answer": "SKU-RED-01"}

    result = run_task(task, EarlyFinalProvider(), "mock", "early-final")
    assert result["success"] is True
    assert result["engine_assisted_tool_calls"] == 2
    assert result["model_initiated_tool_calls"] == 0
    assert result["engine_assisted_completion"] is True
    assert result["model_independent_success"] is False



def test_groq_provider_uses_hosted_chat_api_and_native_tool_calls(monkeypatch):
    import json
    from app.providers import GroqProvider, settings

    captured = []

    class FakeResponse:
        def __init__(self, message):
            self.message = message

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": self.message}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.append({
            "url": url, "headers": headers, "payload": json, "timeout": timeout
        })
        if json.get("tools"):
            return FakeResponse({
                "content": None,
                "tool_calls": [{
                    "id": "call_inventory_1",
                    "type": "function",
                    "function": {
                        "name": "product_lookup",
                        "arguments": '{"sku":"SKU-BLUE-02"}',
                    },
                }],
            })
        return FakeResponse({
            "content": json_module.dumps({
                "type": "final",
                "answer": "60",
                "evidence": "80 less 25 percent equals 60.",
            })
        })

    json_module = json
    monkeypatch.setattr(settings, "groq_api_key", "test-groq-key")
    monkeypatch.setattr("app.providers.httpx.post", fake_post)

    task = next(task for task in TASKS if task["id"] == "math_003")
    action = GroqProvider().next_action(task, [], "openai/gpt-oss-120b")

    assert action == {
        "type": "final",
        "answer": "60",
        "evidence": "80 less 25 percent equals 60.",
    }
    final_payload = captured[-1]["payload"]
    assert captured[-1]["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured[-1]["headers"]["Authorization"] == "Bearer test-groq-key"
    assert final_payload["model"] == "openai/gpt-oss-120b"
    assert final_payload["response_format"]["type"] == "json_schema"
    assert final_payload["response_format"]["json_schema"]["strict"] is True
    assert final_payload["reasoning_effort"] == "low"
    assert final_payload["max_completion_tokens"] == 512

    tool_task = next(task for task in TASKS if task["id"] == "multi_001")
    tool_action = GroqProvider().next_action(
        tool_task, [], "openai/gpt-oss-120b"
    )
    assert tool_action == {
        "type": "tool_call",
        "tool": "product_lookup",
        "arguments": {"sku": "SKU-BLUE-02"},
    }
    tool_payload = captured[-1]["payload"]
    assert tool_payload["tool_choice"] == "required"
    assert tool_payload["parallel_tool_calls"] is False
    assert "response_format" not in tool_payload
    assert tool_payload["tools"][0]["function"]["name"] == "product_lookup"
    assert tool_payload["tools"][0]["function"]["parameters"]["required"] == ["sku"]
    assert "GROQ_API_KEY" not in tool_payload["messages"][1]["content"]


def test_groq_recovery_uses_native_fallback_tool(monkeypatch):
    import json
    from app.providers import GroqProvider, settings

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "call_fallback",
                        "type": "function",
                        "function": {
                            "name": "product_lookup",
                            "arguments": json.dumps({"sku": "SKU-RED-01"}),
                        },
                    }],
                }
            }]}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["payload"] = json
        return FakeResponse()

    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    monkeypatch.setattr("app.providers.httpx.post", fake_post)
    recovery_task = next(task for task in TASKS if task["id"] == "recovery_003")
    action = GroqProvider().next_action(
        recovery_task,
        [{"type": "tool_error", "tool": "product_lookup", "error": "Unknown SKU"}],
        "openai/gpt-oss-120b",
    )
    assert action == {
        "type": "tool_call",
        "tool": "product_lookup",
        "arguments": {"sku": "SKU-RED-01"},
    }
    assert captured["payload"]["tool_choice"] == "required"
    assert "SKU-RED-01" in captured["payload"]["messages"][1]["content"]


def test_groq_provider_rejects_missing_api_key(monkeypatch):
    import pytest
    from app.providers import GroqProvider, settings

    monkeypatch.setattr(settings, "groq_api_key", None)
    task = next(task for task in TASKS if task["id"] == "math_001")
    with pytest.raises(ValueError, match="GROQ_API_KEY is not configured"):
        GroqProvider().next_action(task, [], "openai/gpt-oss-120b")


def test_groq_provider_converts_empty_json_to_retryable_empty_answer(monkeypatch):
    import json
    from app.providers import GroqProvider, settings

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": json.dumps({})}}]}

    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    monkeypatch.setattr(
        "app.providers.httpx.post", lambda *args, **kwargs: FakeResponse()
    )
    task = next(task for task in TASKS if task["id"] == "math_001")
    action = GroqProvider().next_action(task, [], "openai/gpt-oss-120b")
    assert action["type"] == "final"
    assert action["answer"] == ""
    assert "empty action object" in action["evidence"]


def test_groq_retry_prompt_includes_the_rejected_candidate(monkeypatch):
    import json
    from app.providers import GroqProvider, settings

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "type": "final", "answer": "10", "evidence": "Rechecked."
                        })
                    }
                }]
            }

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["payload"] = json
        return FakeResponse()

    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    monkeypatch.setattr("app.providers.httpx.post", fake_post)
    task = next(task for task in TASKS if task["id"] == "logic_001")
    history = [{
        "type": "answer_retry_feedback",
        "candidate_answer": "13",
        "candidate_evidence": "I subtracted from only one group.",
        "message": "Previous answer was rejected.",
    }]
    GroqProvider().next_action(task, history, "openai/gpt-oss-120b")
    prompt = captured["payload"]["messages"][1]["content"]
    assert '"answer": "13"' in prompt
    assert "do not repeat it" in prompt


def test_comparison_ranks_by_model_independent_success_not_pipeline_success(
    monkeypatch, tmp_path
):
    from app import main, storage
    from fastapi.testclient import TestClient

    monkeypatch.setattr(storage, "DATABASE_PATH", tmp_path / "autonomous-ranking.db")

    def fake_run(provider, model):
        pipeline_rate, independent_rate, latency = (
            (1.0, 0.40, 100.0) if model == "assisted-model"
            else (0.75, 0.75, 150.0)
        )
        return {
            "provider": provider,
            "model": model,
            "task_success_rate": pipeline_rate,
            "successful_tasks": round(pipeline_rate * 20),
            "total_tasks": 20,
            "model_independent_task_success_rate": independent_rate,
            "model_independent_successful_tasks": round(independent_rate * 20),
            "engine_assisted_tool_calls": 2 if model == "assisted-model" else 0,
            "average_latency_ms": latency,
            "p95_latency_ms": latency,
            "tool_calls": 2,
            "tool_errors": 0,
            "duplicate_tool_calls": 0,
            "category_metrics": {},
            "results": [],
        }

    monkeypatch.setattr(main, "run", fake_run)
    response = TestClient(main.app).post(
        "/api/benchmark/compare",
        json={
            "provider": "groq",
            "models": ["assisted-model", "autonomous-model"],
            "repetitions": 1,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ranking"][0]["model"] == "autonomous-model"
    assisted = next(
        item for item in data["results"] if item["model"] == "assisted-model"
    )
    assert assisted["mean_task_success_rate"] == 1.0
    assert assisted["mean_model_independent_task_success_rate"] == 0.4


def test_groq_completion_retries_after_tpm_reset_header(monkeypatch):
    import httpx
    from app.providers import groq_chat_completion, settings

    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    monkeypatch.setattr(settings, "groq_max_retries", 2)
    monkeypatch.setattr(settings, "groq_completion_tokens", 512)
    waits = []
    calls = []
    monkeypatch.setattr("app.providers.time.sleep", waits.append)

    request = httpx.Request(
        "POST", "https://api.groq.com/openai/v1/chat/completions"
    )
    rate_limited = httpx.Response(
        429,
        headers={
            "retry-after": "2",
            "x-ratelimit-remaining-tokens": "0",
            "x-ratelimit-reset-tokens": "2s",
        },
        json={
            "error": {
                "message": "Rate limit reached on tokens per minute (TPM).",
                "type": "tokens",
                "code": "rate_limit_exceeded",
            }
        },
        request=request,
    )
    succeeded = httpx.Response(
        200,
        json={"choices": [{"message": {"content": '{"type":"final"}'}}]},
        request=request,
    )

    def fake_post(*args, **kwargs):
        calls.append(kwargs["json"])
        return rate_limited if len(calls) == 1 else succeeded

    monkeypatch.setattr("app.providers.httpx.post", fake_post)
    result = groq_chat_completion(
        "openai/gpt-oss-120b", "system", "user", max_completion_tokens=2048
    )

    assert result == '{"type":"final"}'
    assert len(calls) == 2
    assert waits == [2.0]
    assert calls[0]["max_completion_tokens"] == 512


def test_groq_daily_quota_error_is_not_retried(monkeypatch):
    import httpx
    import pytest
    from app.providers import GroqRateLimitError, groq_chat_completion, settings

    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    calls = []
    request = httpx.Request(
        "POST", "https://api.groq.com/openai/v1/chat/completions"
    )
    response = httpx.Response(
        429,
        headers={"retry-after": "3600"},
        json={
            "error": {
                "message": "Rate limit reached on tokens per day (TPD).",
                "type": "tokens",
                "code": "rate_limit_exceeded",
            }
        },
        request=request,
    )

    def fake_post(*args, **kwargs):
        calls.append(1)
        return response

    monkeypatch.setattr("app.providers.httpx.post", fake_post)
    with pytest.raises(GroqRateLimitError, match="daily Groq quota is exhausted"):
        groq_chat_completion("openai/gpt-oss-120b", "system", "user")
    assert len(calls) == 1


def test_benchmark_stops_and_marks_remaining_tasks_skipped_after_groq_limit(
    monkeypatch,
):
    from app import engine
    from app.providers import GroqRateLimitError

    calls = []

    class RateLimitedProvider:
        def next_action(self, task, history, model):
            calls.append(task["id"])
            raise GroqRateLimitError(
                "Groq rate limit reached (429). tokens per minute.",
                retry_after_seconds=12.0,
            )

    monkeypatch.setattr(engine, "get_provider", lambda name: RateLimitedProvider())
    result = engine.run("groq", "openai/gpt-oss-120b")

    assert calls == [TASKS[0]["id"]]
    assert result["provider_rate_limited"] is True
    assert result["attempted_tasks"] == 1
    assert result["skipped_tasks"] == len(TASKS) - 1
    assert result["total_tasks"] == len(TASKS)
    assert result["p95_latency_ms"] == result["average_latency_ms"]
    assert result["results"][0]["status"] == "rate_limited"
    assert all(
        item["status"] == "skipped_rate_limit"
        for item in result["results"][1:]
    )
