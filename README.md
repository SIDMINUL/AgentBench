# AgentBench

AgentBench is a hosted LLM-agent evaluation platform for reasoning, planning, tool use, and failure recovery. Groq is the default inference provider, so the deployed application does not need a local Ollama runtime or a GPU.

## What it measures

- 22 deterministic benchmark tasks across reasoning, planning, tool use, and failure recovery
- Per-task action traces, including tool arguments, observations, errors, retries, and concise evidence summaries
- Exact-match task success and category-level results
- Model-independent success separated from engine-assisted completion
- Tool-call success, duplicate calls, tool errors, average latency, and p95 latency
- Independent recalculation for selected arithmetic and time tasks; raw model answers are never silently replaced
- Repeated multi-model comparisons (1–5 repetitions per model), saved to SQLite
- Optional Groq-powered reasoning judge; judge scores do not replace benchmark accuracy
- Mock provider for deterministic orchestration tests; optional Ollama provider for local development
- Groq 429 handling that respects retry/reset headers, applies bounded retries, and stops the run cleanly if the available quota is exhausted

## Groq model configuration

The default model is openai/gpt-oss-120b. Groq's model catalog currently lists it as a production model. Model availability and quotas depend on the Groq account and plan; check the current model catalog before changing the model ID.

- Groq API documentation: https://console.groq.com/docs/api-reference
- Groq supported models: https://console.groq.com/docs/models

The Groq API key is read only by the backend from GROQ_API_KEY. It is never sent to the Streamlit browser UI and must not be committed to Git.

The default runtime uses a 512-token completion cap and low reasoning effort to control per-run token consumption. Groq 429 responses are retried only when the API provides a short reset window; daily quota exhaustion stops the benchmark and marks unattempted tasks as skipped.

## Run locally

Use Python 3.11 or newer.

    git clone https://github.com/SIDMINUL/AgentBench.git
    cd AgentBench
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    Copy-Item .env.example .env

Open .env and set your Groq API key:

    GROQ_API_KEY=your_groq_api_key
    DEFAULT_PROVIDER=groq
    DEFAULT_MODEL=openai/gpt-oss-120b
    DEFAULT_JUDGE_MODEL=openai/gpt-oss-120b

Never paste the key into source code, a commit, a screenshot, or a frontend environment variable.

Run the API:

    uvicorn app.main:app --reload

Run the dashboard in another terminal:

    streamlit run dashboard/app.py

Open the dashboard at http://localhost:8501 or API docs at http://localhost:8000/docs.

## Run a benchmark

Groq is the default provider and model:

    python -m app.cli

Or specify them explicitly:

    python -m app.cli --provider groq --model openai/gpt-oss-120b

Validate the orchestration without using an API:

    python -m app.cli --provider mock

Optional local-only Ollama use:

    python -m app.cli --provider ollama --model llama3.2:3b

The mock provider's perfect score validates the test harness only; it is not model performance. For comparisons, use the same task set and at least three repetitions when practical.

## Compare Groq models

Via the Streamlit dashboard, select groq, enter two or more model IDs available to your Groq account, and choose 3 repetitions.

The API equivalent is:

    Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/benchmark/compare -ContentType "application/json" -Body '{"provider":"groq","models":["openai/gpt-oss-120b","qwen/qwen3.8-27b"],"repetitions":3}'

The comparison reports pipeline success and model-independent success separately. Rankings use mean model-independent task success first, then latency. This avoids awarding a model credit for tasks completed through engine-assisted tool calls.

## Deploy on Render

The repository includes a render.yaml Blueprint for an API service and a Streamlit dashboard service.

1. Push the repository to GitHub and create a new Blueprint on Render from SIDMINUL/AgentBench.
2. When prompted, provide GROQ_API_KEY as a secret. Do not put the key in render.yaml.
3. The Blueprint sets DEFAULT_PROVIDER=groq, DEFAULT_MODEL=openai/gpt-oss-120b, and DEFAULT_JUDGE_MODEL=openai/gpt-oss-120b.
4. The API service uses a persistent disk mounted at /var/data; benchmark history is stored at /var/data/agentbench.db.
5. Once both services are live, open the dashboard service URL. Its API URL is wired to the API service over Render's private network.

If you create the services manually instead of using the Blueprint, set these API environment variables:

| Variable | Value |
|---|---|
| GROQ_API_KEY | Your Groq API key (secret) |
| DEFAULT_PROVIDER | groq |
| DEFAULT_MODEL | openai/gpt-oss-120b |
| DEFAULT_JUDGE_MODEL | openai/gpt-oss-120b |
| AGENTBENCH_DB_PATH | /var/data/agentbench.db when using a persistent disk |
| REQUEST_TIMEOUT_SECONDS | 120 |

The dashboard service needs AGENTBENCH_API_URL set to the API service's internal URL, for example http://agentbench-api:10000 when the API service name and port match the Blueprint.

## API endpoints

- GET /api/health
- GET /api/tasks
- POST /api/benchmark/run
- POST /api/benchmark/compare
- POST /api/benchmark/history/{run_id}/judge
- GET /api/benchmark/history?limit=50
- GET /api/benchmark/history/{run_id}

## Benchmark interpretation and limitations

- This is a compact development benchmark, not a standardized or statistically representative evaluation suite.
- Exact-match scoring can mark semantically valid but differently formatted answers as incorrect.
- Model-independent success is the primary metric for autonomous agent claims. Pipeline success can also include tasks finished after engine-assisted required lookups.
- Independent verification covers only tasks that provide a supported structured verification specification. It reports suggestions separately and never overwrites model output.
- The inventory tool uses fixed fixture data, not a live commerce system.
- Latency, model availability, rate limits, and output quality depend on the selected Groq model and account plan.
- A benchmark run records Groq rate-limit headers and waits for short token-window resets when the API supplies a retry window. Daily quota exhaustion is not retried. Remaining tasks are explicitly marked skipped; they are not sent as a burst of doomed requests.
- The agent defaults to a 512-token completion cap and low reasoning effort to keep a 22-task development run within smaller hosted quotas. Override GROQ_COMPLETION_TOKENS and GROQ_REASONING_EFFORT in deployment settings only after checking the limits for your Groq organization.
- Check the Groq Console Limits page for the actual organization counters. Groq limits are organization-scoped and can be request-per-minute/day or token-per-minute/day; a 429 is a rate-limit response, not a model correctness score.
- Repeated runs show variability but do not make this small benchmark statistically representative.
- LLM judge scores are supplementary and should be calibrated against human-reviewed examples before making quality claims.

## Safety

Tools are allowlisted per task. The agent cannot execute arbitrary shell commands or Python code.
