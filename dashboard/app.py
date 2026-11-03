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
        response = httpx.post(
            f"{api_url.rstrip('/')}/api/benchmark/run",
            json={"provider": provider, "model": model},
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        st.success(f'Run #{data["run_id"]} saved.')
        first, second, third, fourth = st.columns(4)
        first.metric("Task success", f'{data["task_success_rate"]:.1%}')
        second.metric("Average latency", f'{data["average_latency_ms"]:.1f} ms')
        third.metric("Successful tasks", f'{data["successful_tasks"]}/{data["total_tasks"]}')
        fourth.metric("Tool calls", data["tool_calls"])
        st.subheader("Task results")
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
st.caption("Runs the same task set sequentially for every selected model. Each run is saved to history.")
comparison_models = st.text_input(
    "Ollama model names (comma-separated)",
    "llama3.2:3b, llama3.1:8b",
)
if st.button("Compare models"):
    models = [item.strip() for item in comparison_models.split(",") if item.strip()]
    if len(models) < 2:
        st.warning("Enter at least two distinct model names.")
    elif len({item.casefold() for item in models}) != len(models):
        st.warning("Model names must be unique.")
    else:
        try:
            with st.spinner("Running the same benchmark tasks against each model..."):
                response = httpx.post(
                    f"{api_url.rstrip('/')}/api/benchmark/compare",
                    json={"provider": provider, "models": models},
                    timeout=900,
                )
            response.raise_for_status()
            comparison = response.json()
            st.success(f'{comparison["comparison_count"]} model runs completed and saved.')
            comparison_df = pd.DataFrame(comparison["ranking"])
            comparison_df["task_success_rate"] = comparison_df["task_success_rate"].map(lambda value: f"{value:.1%}")
            comparison_df["tool_call_success_rate"] = comparison_df["tool_call_success_rate"].map(lambda value: f"{value:.1%}")
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            st.caption(comparison["ranking_rule"])
            st.bar_chart(
                pd.DataFrame(comparison["results"]).set_index("model")["task_success_rate"]
            )
        except Exception as exc:
            st.error(f"Model comparison failed: {exc}")

st.divider()
st.subheader("Benchmark run history")
try:
    history_response = httpx.get(
        f"{api_url.rstrip('/')}/api/benchmark/history",
        params={"limit": 100},
        timeout=15,
    )
    history_response.raise_for_status()
    runs = history_response.json()["runs"]
    if runs:
        history_df = pd.DataFrame(runs)
        history_df["task_success_rate"] = history_df["task_success_rate"].map(lambda value: f"{value:.1%}")
        history_df["average_latency_ms"] = history_df["average_latency_ms"].map(lambda value: round(value, 1))
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        selected_run = st.number_input("Inspect run ID", min_value=1, step=1)
        if st.button("Load selected run"):
            detail = httpx.get(
                f"{api_url.rstrip('/')}/api/benchmark/history/{int(selected_run)}",
                timeout=15,
            )
            if detail.status_code == 404:
                st.warning("That run ID was not found.")
            else:
                detail.raise_for_status()
                st.json(detail.json())
    else:
        st.write("No saved benchmark runs yet. Run a benchmark to create your first record.")
except Exception as exc:
    st.warning(f"Could not load benchmark history: {exc}")
