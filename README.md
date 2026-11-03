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
- Multi-model comparison using an identical task set
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

## Run a benchmark
```bash
python -m app.cli --provider mock
ollama pull llama3.2:3b
python -m app.cli --provider ollama --model llama3.2:3b
```

## Compare models
Send at least two model names to the comparison endpoint:
```bash
curl -X POST http://127.0.0.1:8000/api/benchmark/compare ^
  -H "Content-Type: application/json" ^
  -d "{\"provider\":\"ollama\",\"models\":[\"llama3.2:3b\",\"llama3.1:8b\"]}"
```
The dashboard also provides a model comparison panel. Every model is evaluated against the same task set, and each individual run is saved in SQLite. Results are ranked by task success rate, with average latency used to break ties. Model execution is sequential to avoid competing for local GPU memory.

Each CLI or API run is saved to `data/agentbench.db` by default. Override the location with `AGENTBENCH_DB_PATH`.

## API endpoints
- `GET /api/health` — service health
- `GET /api/tasks` — benchmark task definitions
- `POST /api/benchmark/run` — execute and persist a benchmark
- `POST /api/benchmark/compare` — compare 2–8 models on the same tasks
- `GET /api/benchmark/history?limit=50` — list recent runs
- `GET /api/benchmark/history/{run_id}` — retrieve a complete run and its traces

## Evaluation note
Mock results validate the evaluation pipeline only; they are not evidence of model performance. Exact-match scores are intentionally strict and can mark semantically correct but differently phrased answers as incorrect. A comparison is only meaningful when the same task definitions and provider settings are used for every model.

## Safety
Agent tools are allowlisted per task. The agent cannot execute arbitrary shell commands or Python code.
