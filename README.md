# AgentBench

A reproducible platform for evaluating LLM agents on multi-step tasks, tool use, and final-answer quality.

## Features
- Mock and Ollama model providers
- Multi-step agent execution with allowlisted tools and execution traces
- Exact-match task scoring, tool-call success, average and p95 latency
- FastAPI API and Streamlit dashboard
- SQLite persistence for CLI and API benchmark runs
- Multi-model comparison using identical task sets
- Optional Ollama-based LLM-as-a-Judge scoring
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
```bash
python -m app.cli --provider mock
ollama pull llama3.2:3b
python -m app.cli --provider ollama --model llama3.2:3b
```

## Compare models
Use `POST /api/benchmark/compare` with at least two model names:
```json
{"provider":"ollama","models":["llama3.2:3b","qwen2.5:3b"]}
```
The dashboard provides the same workflow. Runs execute sequentially and are saved to SQLite.

## LLM-as-a-Judge
Use `POST /api/benchmark/history/{run_id}/judge` with:
```json
{"judge_model":"llama3.2:3b"}
```
The judge scores correctness, reasoning quality, tool-use quality, and overall quality from 1–5, with a short rationale per task. Judgments are stored with the run and never overwrite exact-match metrics. These model-generated scores are supplementary, not ground truth.

## Persistence and API
Runs are saved to `data/agentbench.db` by default. Set `AGENTBENCH_DB_PATH` to change the location.
- `GET /api/health`
- `GET /api/tasks`
- `POST /api/benchmark/run`
- `POST /api/benchmark/compare`
- `POST /api/benchmark/history/{run_id}/judge`
- `GET /api/benchmark/history?limit=50`
- `GET /api/benchmark/history/{run_id}`

## Evaluation notes
Mock results validate orchestration only, not model performance. Exact-match scoring can mark semantically correct but differently phrased answers as incorrect. LLM-as-a-Judge should be calibrated against human-reviewed examples before being treated as reliable.

## Safety
Agent tools are allowlisted per task. The agent cannot execute arbitrary shell commands or Python code.
