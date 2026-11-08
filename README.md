# AgentBench

AgentBench is a hosted LLM-agent evaluation platform for reasoning, planning, tool use, and failure recovery. Groq is the default inference provider, so the deployed application does not need a local Ollama runtime or a GPU.

## What it measures

- 22 deterministic benchmark tasks across reasoning, planning, tool use, and failure recovery
- Per-task action traces, including tool arguments, observations, errors, retries, and concise evidence summaries
- Groq-native function calling for inventory tools, with tool schemas and explicit required-tool selection; tool results are executed locally through the allowlisted registry
- Exact-match task success and category-level results
- Model-independent success separated from engine-assisted completion
- Tool-call success, duplicate calls, tool errors, average latency, and p95 latency
- Independent recalculation for selected arithmetic and time tasks; raw model answers are never silently replaced
- Multi-model comparisons (1–5 repetitions per model) saved to SQLite, with a quick one-run default and up to two model runs executed concurrently
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

Open .env and set your Groq API key. For local development, APP_ACCESS_PASSWORD can remain blank; deployed Render services require it.

    GROQ_API_KEY=your_groq_api_key
    APP_ACCESS_PASSWORD=
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

The mock provider's perfect score validates the test harness only; it is not model performance. The comparison UI defaults to one run per model for a quick check. Use three or more repetitions when you need a more reliable estimate; these take longer.

## Compare Groq models

Via the Streamlit dashboard, select groq, enter two or more model IDs available to your Groq account, and choose 1 repetition for a quick comparison or 3+ for a more reliable but slower comparison. Independent model runs execute concurrently, capped at two by default to avoid a large burst of Groq requests.

The API equivalent is:

    Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/benchmark/compare -ContentType "application/json" -Body '{"provider":"groq","models":["openai/gpt-oss-120b","qwen/qwen3.8-27b"],"repetitions":3}'

The comparison reports pipeline success and model-independent success separately. Rankings use mean model-independent task success first, then latency. This avoids awarding a model credit for tasks completed through engine-assisted tool calls.

## Deploy a free, password-protected demo on Render

The Blueprint provisions two **Free** Render web services in Singapore:

- `agentbench-api`: FastAPI + Groq, Free plan
- `agentbench-dashboard`: Streamlit UI, Free plan

The first Blueprint sync prompts for two secrets:

1. `GROQ_API_KEY`: your Groq API key
2. `APP_ACCESS_PASSWORD`: choose a strong shared password for the portfolio demo

Do not commit either secret. The dashboard asks visitors for APP_ACCESS_PASSWORD and sends it to the API over HTTPS using HTTP Basic authentication. The API protects all endpoints except its public health check.

The public demo has conservative per-IP controls: up to 3 single benchmark runs per hour, 1 model comparison per hour, and 3 judge requests per hour. A comparison is capped at 2 models and 2 repetitions. These controls reduce accidental Groq quota exhaustion; they are not a substitute for a managed identity provider or distributed abuse protection.

### Free-plan limitations

This configuration is intended for a portfolio/demo environment, not production:

- Render Free web services spin down after 15 minutes without traffic and can take about a minute to wake up.
- Both services share your workspace's 750 Free instance hours per calendar month.
- Free web services have ephemeral filesystems. The SQLite run history can disappear when the API service spins down, restarts, or redeploys. This free configuration intentionally does not provision a paid persistent disk or paid database.
- Free web services cannot receive private-network requests. The dashboard therefore calls the API through its public HTTPS URL.
- Free bandwidth and build-pipeline allowances apply. Monitor usage in Render's Billing page; keep the workspace spend limit at $0 and avoid adding a paid service.
- Groq Free usage is subject to model-specific organization limits. When a limit is reached, AgentBench stops or skips work instead of switching to paid usage.

See Render's current free-service limitations: https://render.com/docs/free
See Groq's current rate limits: https://console.groq.com/docs/rate-limits

### Deploy steps

1. Push the repository's `main` branch to GitHub.
2. In Render, choose **New → Blueprint** and connect `SIDMINUL/AgentBench`.
3. Review the plan. It should contain two Free web services only and no persistent disk or database.
4. Supply `GROQ_API_KEY` and your chosen `APP_ACCESS_PASSWORD` when prompted.
5. Apply the Blueprint. Open the `agentbench-dashboard` URL and enter the shared password in the sidebar.

If Render's plan preview shows any paid resource, do not apply it. The intended demo setup should cost $0 on Render while usage stays within the included Free allowances, and $0 on Groq while the account remains on its Free tier.

For a durable run history, always-on availability, higher limits, or public production use, upgrade the relevant hosting/database/API plan deliberately after reviewing the costs.

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
