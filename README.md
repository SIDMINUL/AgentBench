# AgentBench

A reproducible platform for evaluating LLM agents on multi-step tasks, tool use, and final-answer quality.

## Phase 1
- Sample benchmark tasks
- Mock and Ollama model providers
- Deterministic answer checks and latency metrics
- FastAPI endpoints
- Streamlit dashboard
- Basic automated test

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

Run a deterministic pipeline check:
```bash
python -m app.cli --provider mock
```

For Ollama:
```bash
ollama pull llama3.2:3b
python -m app.cli --provider ollama --model llama3.2:3b
```

Mock results validate the evaluation pipeline only; they are not evidence of model performance.

## Planned next
Agent tool-calling loop, execution traces, tool-call validation, LLM-as-a-judge, PostgreSQL persistence, model comparisons, and deployment.

## Safety
The project will use allowlisted tools. It will not expose arbitrary shell or Python execution to agents.
