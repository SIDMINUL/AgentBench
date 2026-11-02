from app.engine import run
from app.tasks import TASKS
from app.providers import MockProvider
from app.engine import run_task

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
    result = run_task(task, BadProvider(), "test", "bad")
    assert result["success"] is False
    assert "Tool not allowed" in result["error"]
