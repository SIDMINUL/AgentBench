import os

import httpx
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AgentBench", layout="wide")
st.title("AgentBench")
st.caption(
    "Hosted LLM agent evaluation. Groq is the deployment provider; "
    "benchmark history is stored by the API service."
)

default_api_url = os.getenv("AGENTBENCH_API_URL", "http://localhost:8000")
if not default_api_url.startswith(("http://", "https://")):
    default_api_url = f"http://{default_api_url}"
api_url = st.sidebar.text_input("API URL", default_api_url)
provider = st.sidebar.selectbox(
    "Provider",
    ["groq", "mock", "ollama"],
    help="Use Groq for hosted inference. Mock validates the pipeline only.",
)
model_defaults = {
    "groq": os.getenv("DEFAULT_MODEL", "openai/gpt-oss-120b"),
    "mock": "mock-agent",
    "ollama": os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
}
model = st.sidebar.text_input("Model ID", model_defaults[provider])

st.header("Single-model benchmark")
if st.button("Run benchmark", type="primary"):
    try:
        response = httpx.post(
            f"{api_url.rstrip('/')}/api/benchmark/run",
            json={"provider": provider, "model": model},
            timeout=1800,
        )
        response.raise_for_status()
        data = response.json()
        st.success(f'Run #{data["run_id"]} saved.')
        a, b, c, d, e = st.columns(5)
        a.metric("Pipeline success", f'{data["task_success_rate"]:.1%}')
        b.metric(
            "Model-independent success",
            f'{data.get("model_independent_task_success_rate", data["task_success_rate"]):.1%}',
        )
        c.metric("Successful tasks", f'{data["successful_tasks"]}/{data["total_tasks"]}')
        d.metric("Average latency", f'{data["average_latency_ms"]:.1f} ms')
        e.metric("Engine-assisted calls", data.get("engine_assisted_tool_calls", 0))
        st.dataframe(pd.DataFrame(data["results"]), use_container_width=True)
        with st.expander("Execution traces and evidence"):
            for item in data["results"]:
                st.markdown(f'**{item["task_id"]}**')
                if item.get("evidence"):
                    st.caption(f'Evidence: {item["evidence"]}')
                st.json(item["trace"])
        if provider == "mock":
            st.info("Mock results validate orchestration only, not model capability.")
    except Exception as exc:
        st.error(f"Benchmark request failed: {exc}")

st.divider()
st.header("Compare models")
st.caption(
    "Every model receives the same task set. Ranking uses model-independent "
    "success, so engine-assisted completions do not improve a model's rank."
)
comparison_default = (
    "openai/gpt-oss-120b, qwen/qwen3.8-27b"
    if provider == "groq"
    else ("mock-agent, mock-agent-2" if provider == "mock"
          else "qwen2.5:3b, llama3.2:3b")
)
comparison_models = st.text_input(
    "Model IDs (comma-separated)",
    comparison_default,
)
repetitions = st.number_input(
    "Runs per model", min_value=1, max_value=5, value=3, step=1
)
if st.button("Compare models"):
    models = [item.strip() for item in comparison_models.split(",") if item.strip()]
    if len(models) < 2 or len({item.casefold() for item in models}) != len(models):
        st.warning("Enter at least two distinct model IDs.")
    else:
        try:
            with st.spinner("Running identical benchmark tasks for each model..."):
                response = httpx.post(
                    f"{api_url.rstrip('/')}/api/benchmark/compare",
                    json={
                        "provider": provider,
                        "models": models,
                        "repetitions": int(repetitions),
                    },
                    timeout=3600,
                )
            response.raise_for_status()
            comparison = response.json()
            st.success(
                f'{comparison["total_runs"]} runs completed across '
                f'{comparison["comparison_count"]} models.'
            )
            comparison_rows = []
            for item in comparison["ranking"]:
                row = dict(item)
                category_metrics = row.pop("category_metrics", {})
                row["run_ids"] = ", ".join(
                    str(value) for value in row.get("run_ids", [])
                )
                row["pipeline_success_rate"] = row.pop("mean_task_success_rate")
                row["autonomous_success_rate"] = row.get(
                    "mean_model_independent_task_success_rate", 0.0
                )
                row["success_rate_range"] = "{:.1%}–{:.1%}".format(
                    row.get("success_rate_min", 0.0),
                    row.get("success_rate_max", 0.0),
                )
                for category in (
                    "reasoning", "planning", "tool_use", "failure_recovery"
                ):
                    metric = category_metrics.get(category, {})
                    row[f"{category}_success"] = (
                        f'{metric.get("successful_tasks", 0)}/'
                        f'{metric.get("total_tasks", 0)}'
                    )
                    row[f"{category}_rate"] = (
                        f'{metric.get("task_success_rate", 0.0):.1%}'
                    )
                comparison_rows.append(row)

            df = pd.DataFrame(comparison_rows)
            for column in ("pipeline_success_rate", "autonomous_success_rate",
                           "tool_call_success_rate"):
                if column in df:
                    df[column] = df[column].map(lambda value: f"{value:.1%}")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(comparison["ranking_rule"])
            chart = pd.DataFrame(comparison["ranking"]).set_index("model")
            st.bar_chart(chart["mean_model_independent_task_success_rate"])
        except Exception as exc:
            st.error(f"Model comparison failed: {exc}")

st.divider()
st.header("LLM-as-a-Judge")
st.caption("Groq judge scores are supplementary; they do not replace exact-match metrics.")
judge_model = st.text_input(
    "Judge model",
    os.getenv("DEFAULT_JUDGE_MODEL", "openai/gpt-oss-120b"),
)
judge_run_id = st.number_input("Run ID to evaluate", min_value=1, step=1)
if st.button("Evaluate saved run"):
    try:
        response = httpx.post(
            f"{api_url.rstrip('/')}/api/benchmark/history/{int(judge_run_id)}/judge",
            json={"judge_model": judge_model},
            timeout=600,
        )
        response.raise_for_status()
        judged = response.json()
        st.success(f'Average overall score: {judged["average_overall_score"]:.2f}/5')
        st.dataframe(
            pd.DataFrame(judged["evaluations"]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(judged["note"])
    except Exception as exc:
        st.error(f"Judging failed: {exc}")

st.divider()
st.subheader("Benchmark run history")
try:
    response = httpx.get(
        f"{api_url.rstrip('/')}/api/benchmark/history",
        params={"limit": 100},
        timeout=30,
    )
    response.raise_for_status()
    runs = response.json()["runs"]
    if runs:
        df = pd.DataFrame(runs)
        df["task_success_rate"] = df["task_success_rate"].map(
            lambda value: f"{value:.1%}"
        )
        if "model_independent_task_success_rate" in df:
            df["model_independent_task_success_rate"] = df[
                "model_independent_task_success_rate"
            ].map(lambda value: f"{value:.1%}")
        df["average_latency_ms"] = df["average_latency_ms"].map(
            lambda value: round(value, 1)
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
        selected = st.number_input(
            "Inspect run ID", min_value=1, step=1, key="inspect_run_id"
        )
        if st.button("Load selected run"):
            detail = httpx.get(
                f"{api_url.rstrip('/')}/api/benchmark/history/{int(selected)}",
                timeout=30,
            )
            if detail.status_code == 404:
                st.warning("That run ID was not found.")
            else:
                detail.raise_for_status()
                st.json(detail.json())
    else:
        st.write("No saved benchmark runs yet.")
except Exception as exc:
    st.warning(f"Could not load benchmark history: {exc}")
