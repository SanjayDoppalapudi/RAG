import asyncio
from pathlib import Path
import time

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests
import viz_utils
from data_loader import embed_texts
from vector_db import QdrantStorage

load_dotenv()

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path


async def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )


st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    with st.spinner("Uploading and triggering ingestion..."):
        path = save_uploaded_pdf(uploaded)
        # Kick off the event and block until the send completes
        asyncio.run(send_rag_ingest_event(path))
        # Small pause for user feedback continuity
        time.sleep(0.3)
    st.success(f"Triggered ingestion for: {path.name}")
    st.caption("You can upload another PDF if you like.")

st.divider()
st.title("Ask a question about your PDFs")


async def send_rag_query_event(question: str, top_k: int) -> None:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )

    return result[0]


def _inngest_api_base() -> str:
    # Local dev server default; configurable via env
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)


with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Sending event and generating answer..."):
            # Fire-and-forget event to Inngest for observability/workflow
            event_id = asyncio.run(send_rag_query_event(question.strip(), int(top_k)))
            # Poll the local Inngest API for the run's output
            output = wait_for_run_output(event_id)
            answer = output.get("answer", "")
            sources = output.get("sources", [])

        st.subheader("Answer")
        st.write(answer or "(No answer)")
        if sources:
            st.caption("Sources")
            for s in sources:
                st.write(f"- {s}")

st.divider()
st.title("Semantic Space Visualization")

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Visualize 3D Vector Space"):
        with st.spinner("Calculating vector space..."):
            query_vec = None
            if question and question.strip():
                try:
                    # Embed the question locally to visualize it relative to the docs
                    query_vec = embed_texts([question])[0]
                except Exception as e:
                    st.warning(f"Could not embed query: {e}")

            df, fig = viz_utils.get_visualization_data(query_vector=query_vec, query_text=question)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Visualizing {len(df)} points. Rotate to explore clusters.")
            else:
                st.warning("No data found in vector database.")

st.divider()
st.subheader("Manage Sources")

# Helper to list sources (uncached or short TTL to reflect updates)
def get_current_sources():
    try:
        with QdrantStorage() as store:
            return store.list_sources()
    except Exception as e:
        st.error(f"Database error: {e}")
        return []

sources_list = get_current_sources()

if sources_list:
    col_del1, col_del2 = st.columns([3, 1])
    with col_del1:
        to_delete = st.selectbox("Select a source to delete", sources_list, key="delete_source_select")
    with col_del2:
        st.write("") # Spacer
        st.write("") # Spacer
        if st.button("Delete Source", type="primary", key="delete_source_btn"):
            if to_delete:
                try:
                    with st.spinner(f"Deleting {to_delete}..."):
                        with QdrantStorage() as store:
                            store.delete_source(to_delete)
                        time.sleep(0.5) # Give DB a moment
                    st.success(f"Deleted {to_delete}")
                    st.rerun()
                except Exception as e:
                     st.error(f"Could not delete source (DB locked?). Try again. {e}")
else:
    st.info("No sources found in the database.")
