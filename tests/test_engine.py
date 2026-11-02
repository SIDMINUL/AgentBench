from app.engine import run

def test_mock_benchmark_passes():
    result = run("mock", "mock-agent")
    assert result["total_tasks"] == 3
    assert result["task_success_rate"] == 1.0
