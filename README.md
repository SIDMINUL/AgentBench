# AgentBench

A reproducible platform for evaluating LLM agents on multi-step tasks, tool use, and final-answer quality.

## Features
- Mock and Ollama model providers
- Multi-step agent execution with allowlisted tools
- Per-step execution traces and tool error tracking
- Exact-match task scoring, tool-call success, average and p95 latency
- FastAPI benchmark API
- Streamlit dashboard with saved benchmark history
- SQLite persistence for CLI and API benchmark runs
- Automated tests

## Quick start
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs: http://127.0.0.1:8000/docs

In another terminal:
```bash
streamlit run dashboard/app.py
```

## Run benchmarks
Deterministic pipeline check:
```bash
python -m app.cli --provider mock
```

Ollama:
```bash
ollama pull llama3.2:3b
python -m app.cli --provider ollama --model llama3.2:3b
```

Each CLI or API run is saved to `data/agentbench.db` by default. Override the location with `AGENTBENCH_DB_PATH`.

## API endpoints
- `GET /api/health` — service health
- `GET /api/tasks` — benchmark task definitions
- `POST /api/benchmark/run` — execute and persist a benchmark
- `GET /api/benchmark/history?limit=50` — list recent runs
- `GET /api/benchmark/history/{run_id}` — retrieve a complete run and its traces

## Evaluation note
Mock results validate the evaluation pipeline only; they are not evidence of model performance. Exact-match scores are intentionally strict and can mark semantically correct but differently phrased answers as incorrect.

## Safety
Agent tools are allowlisted per task. The agent cannot execute arbitrary shell commands or Python code.
