import httpx
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AgentBench", layout="wide")
st.title("AgentBench")
st.caption("LLM Agent Evaluation Platform")

api_url = st.sidebar.text_input("API URL", "http://localhost:8000")
provider = st.sidebar.selectbox("Provider", ["mock", "ollama"])
model = st.sidebar.text_input("Model", "llama3.2:3b" if provider == "ollama" else "mock-agent")

if st.button("Run benchmark", type="primary"):
    try:
        response = httpx.post(
            f"{api_url.rstrip('/')}/api/benchmark/run",
            json={"provider": provider, "model": model},
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        first, second, third = st.columns(3)
        first.metric("Task success", f'{data["task_success_rate"]:.1%}')
        second.metric("Average latency", f'{data["average_latency_ms"]:.1f} ms')
        third.metric("Successful tasks", f'{data["successful_tasks"]}/{data["total_tasks"]}')
        st.subheader("Task results")
        st.dataframe(pd.DataFrame(data["results"]), use_container_width=True)
        if provider == "mock":
            st.info("Mock results validate the pipeline only, not model capability.")
    except Exception as exc:
        st.error(f"Benchmark request failed: {exc}")
else:
    st.write("Run the benchmark to view task-level results.")
