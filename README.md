# AgentBench

A reproducible platform for evaluating LLM agents on reasoning, planning, multi-step tool use, and failure recovery.

## Features

- Mock and Ollama model providers
- Multi-step agent execution with per-task tool allowlists and required-observation guards
- Deterministic execution of task-declared mandatory and fallback lookups when the model attempts a premature final answer; engine-assisted calls are separately traced and counted
- Execution traces containing tool arguments, observations, errors, and final answers
- Deterministic exact-match task success, category-level success rates, tool-call success, average latency, and p95 latency
- Independent deterministic verification for selected arithmetic/time tasks, preserving raw model answers and reporting suggested corrections separately
- 22 benchmark tasks across reasoning, planning, tool use, and three failure-recovery scenarios
- FastAPI API and Streamlit dashboard
- SQLite persistence for CLI and API benchmark runs
- Multi-model comparison using identical task sets
- Optional Ollama-based LLM-as-a-Judge
- Automated tests for orchestration, tool use, multi-tool workflows, and recovery

## Benchmark task coverage

| Category | Tasks | What is tested |
|---|---:|---|
| Reasoning | 8 | Arithmetic, percentages, ratios, relational ordering, and logical inference |
| Planning | 6 | Multi-stage dependencies and time-based scheduling, including breaks |
| Tool use | 5 | Inventory lookups, stock thresholds, conditional decisions, and multiple tool calls |
| Failure recovery | 3 | Recovering from invalid SKUs using different fallback products and stock conditions |
| **Total** | **22** | |

The benchmark uses a small deterministic inventory tool with known fixture data. Three failure-recovery tasks intentionally request unknown SKUs and require different valid fallback lookups. The execution engine blocks premature final answers until declared required tool observations (including both lookups in the multi-product comparison) and any required recovery lookup have succeeded. Run summaries report success rates separately for reasoning, planning, tool use, and failure recovery.

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

The mock provider is a deterministic fixture for testing the benchmark engine. Its perfect score does not represent real model performance. Use an Ollama model to measure actual agent behavior.

## Compare models

Use `POST /api/benchmark/compare` with at least two model names:

```json
{"provider":"ollama","models":["qwen2.5:3b","llama3.2:3b","llama3:latest","llama2:latest"],"repetitions":3}
```

The dashboard provides the same workflow. The default comparison includes Qwen 2.5 3B, Llama 3.2 3B, Llama 3, and Llama 2; verify that each model is installed locally with `ollama list` before running. Each model runs the same task set the requested number of times (1–5, default 1), and every individual run is saved to SQLite. Results report mean success rate, minimum/maximum success rate, mean latency, and aggregated category metrics. Rank by mean success rate, then mean latency.

## LLM-as-a-Judge

Use `POST /api/benchmark/history/{run_id}/judge` with:

```json
{"judge_model":"llama3.2:3b"}
```

The scoring dimensions are separated to reduce judge-model errors:

- **Correctness:** derived from the benchmark's deterministic success flag.
- **Reasoning quality:** assessed by the LLM judge on a 1–5 scale.
- **Tool-use quality:** calculated from task tool permissions and the recorded execution trace.
- **Overall:** arithmetic mean of correctness, reasoning, and tool-use scores.

LLM rationales and reasoning scores are supplementary assessments, not ground truth. Review samples manually before using judge scores for formal model claims.

## Persistence and API

Runs are saved to `data/agentbench.db` by default. Set `AGENTBENCH_DB_PATH` to change the location.

- `GET /api/health`
- `GET /api/tasks`
- `POST /api/benchmark/run`
- `POST /api/benchmark/compare`
- `POST /api/benchmark/history/{run_id}/judge`
- `GET /api/benchmark/history?limit=50`
- `GET /api/benchmark/history/{run_id}`

## Evaluation notes and limitations

- The included tasks are a compact development benchmark, not a standardized or statistically representative evaluation suite. The engine enforces explicitly declared recovery requirements; report this orchestration policy when comparing agent performance.
- Mock results validate orchestration only, not model capability.
- Exact-match scoring can mark semantically correct but differently phrased answers as incorrect.
- The inventory tool uses fixed fixture data; it does not connect to a live commerce system.
- Latency depends on local hardware, model size, warm-up state, and Ollama configuration.
- Selected arithmetic/time tasks have a deterministic verifier that recomputes answers from structured task data. It records the raw model answer, reference answer, and any suggested correction separately; it never overwrites the model output or changes the raw task-success score.
- The verifier currently covers five templates: count after removal, fraction then return, elapsed time, percentage discount, and proportional ratio. Other tasks are not independently verified by this mechanism.
- Category-level metrics help isolate weaknesses, but the reasoning category currently contains eight tasks and should not be treated as a statistically robust score.
- For meaningful comparisons, use at least three repetitions when practical, report the observed success-rate range, and keep task prompts and tool fixtures identical. Repetitions improve visibility into run-to-run variability but do not make this compact task suite statistically representative.
- LLM-as-a-Judge should be calibrated against human-reviewed examples before being treated as reliable.

## Safety

Agent tools are allowlisted per task. The agent cannot execute arbitrary shell commands or Python code.
