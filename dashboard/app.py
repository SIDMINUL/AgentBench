import httpx
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AgentBench", layout="wide")
st.title("AgentBench")
st.caption("LLM Agent Evaluation Platform — benchmark history is saved locally.")
api_url = st.sidebar.text_input("API URL", "http://localhost:8000")
provider = st.sidebar.selectbox("Provider", ["mock", "ollama"])
model = st.sidebar.text_input("Model", "llama3.2:3b" if provider == "ollama" else "mock-agent")

st.header("Single-model benchmark")
if st.button("Run benchmark", type="primary"):
    try:
        response = httpx.post(f"{api_url.rstrip('/')}/api/benchmark/run",
            json={"provider": provider, "model": model}, timeout=300)
        response.raise_for_status()
        data = response.json()
        st.success(f'Run #{data["run_id"]} saved.')
        a, b, c, d = st.columns(4)
        a.metric("Task success", f'{data["task_success_rate"]:.1%}')
        b.metric("Average latency", f'{data["average_latency_ms"]:.1f} ms')
        c.metric("Successful tasks", f'{data["successful_tasks"]}/{data["total_tasks"]}')
        d.metric("Tool calls", data["tool_calls"])
        st.dataframe(pd.DataFrame(data["results"]), use_container_width=True)
        with st.expander("Execution traces"):
            for item in data["results"]:
                st.markdown(f'**{item["task_id"]}**')
                st.json(item["trace"])
        if provider == "mock":
            st.info("Mock results validate the pipeline only, not model capability.")
    except Exception as exc:
        st.error(f"Benchmark request failed: {exc}")

st.divider()
st.header("Compare models")
st.caption("Runs identical tasks sequentially; every model run is saved.")
comparison_models = st.text_input("Ollama model names (comma-separated)", "qwen2.5:3b, llama3.2:3b, llama3:latest, llama2:latest")
if st.button("Compare models"):
    models = [item.strip() for item in comparison_models.split(",") if item.strip()]
    if len(models) < 2 or len({item.casefold() for item in models}) != len(models):
        st.warning("Enter at least two distinct model names.")
    else:
        try:
            with st.spinner("Running benchmark against each model..."):
                response = httpx.post(f"{api_url.rstrip('/')}/api/benchmark/compare",
                    json={"provider": provider, "models": models}, timeout=900)
            response.raise_for_status()
            comparison = response.json()
            st.success(f'{comparison["comparison_count"]} model runs completed and saved.')
            comparison_rows = []
            for item in comparison["ranking"]:
                row = dict(item)
                category_metrics = row.pop("category_metrics", {})
                for category in ("reasoning", "planning", "tool_use", "failure_recovery"):
                    metric = category_metrics.get(category, {})
                    row[f"{category}_success"] = (
                        f'{metric.get("successful_tasks", 0)}/{metric.get("total_tasks", 0)}'
                    )
                    row[f"{category}_rate"] = f'{metric.get("task_success_rate", 0.0):.1%}'
                comparison_rows.append(row)
            df = pd.DataFrame(comparison_rows)
            df["task_success_rate"] = df["task_success_rate"].map(lambda value: f"{value:.1%}")
            df["tool_call_success_rate"] = df["tool_call_success_rate"].map(lambda value: f"{value:.1%}")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(comparison["ranking_rule"])
            st.bar_chart(pd.DataFrame(comparison["results"]).set_index("model")["task_success_rate"])
        except Exception as exc:
            st.error(f"Model comparison failed: {exc}")

st.divider()
st.header("LLM-as-a-Judge")
st.caption("Supplementary quality scores; these do not replace exact-match metrics.")
judge_model = st.text_input("Judge model", "llama3.2:3b")
judge_run_id = st.number_input("Run ID to evaluate", min_value=1, step=1)
if st.button("Evaluate saved run"):
    try:
        response = httpx.post(f"{api_url.rstrip('/')}/api/benchmark/history/{int(judge_run_id)}/judge",
            json={"judge_model": judge_model}, timeout=300)
        response.raise_for_status()
        judged = response.json()
        st.success(f'Average overall score: {judged["average_overall_score"]:.2f}/5')
        st.dataframe(pd.DataFrame(judged["evaluations"]), use_container_width=True, hide_index=True)
        st.caption(judged["note"])
    except Exception as exc:
        st.error(f"Judging failed: {exc}")

st.divider()
st.subheader("Benchmark run history")
try:
    response = httpx.get(f"{api_url.rstrip('/')}/api/benchmark/history",
        params={"limit": 100}, timeout=15)
    response.raise_for_status()
    runs = response.json()["runs"]
    if runs:
        df = pd.DataFrame(runs)
        df["task_success_rate"] = df["task_success_rate"].map(lambda value: f"{value:.1%}")
        df["average_latency_ms"] = df["average_latency_ms"].map(lambda value: round(value, 1))
        st.dataframe(df, use_container_width=True, hide_index=True)
        selected = st.number_input("Inspect run ID", min_value=1, step=1, key="inspect_run_id")
        if st.button("Load selected run"):
            detail = httpx.get(f"{api_url.rstrip('/')}/api/benchmark/history/{int(selected)}", timeout=15)
            if detail.status_code == 404:
                st.warning("That run ID was not found.")
            else:
                detail.raise_for_status()
                st.json(detail.json())
    else:
        st.write("No saved benchmark runs yet.")
except Exception as exc:
    st.warning(f"Could not load benchmark history: {exc}")
