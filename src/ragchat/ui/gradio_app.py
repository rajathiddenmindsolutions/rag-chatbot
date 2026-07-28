"""Gradio UI application — streaming version calling FastAPI over HTTP + SSE."""

import re
import json
from pathlib import Path

import httpx
import gradio as gr

API_URL = "http://localhost:8000/api"


# ─────────────────────────────────────────────────────────────────────────────
# Backend helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_ingested_documents():
    """Fetch list of ingested documents from FastAPI."""
    try:
        response = httpx.get(f"{API_URL}/documents", timeout=10.0)
        if response.status_code == 200:
            docs = response.json()
            if not docs:
                return "No documents ingested yet."

            md = "### Ingested Documents\n\n"
            md += "| Title | Authors | Status | Ingested At |\n"
            md += "| :--- | :--- | :--- | :--- |\n"
            for d in docs:
                title = d.get("title") or Path(d["source_path"]).name
                authors = ", ".join(d.get("authors") or []) or "Unknown"
                md += f"| {title} | {authors} | `{d['status']}` | {d['ingested_at'][:19]} |\n"
            return md
        else:
            return f"Error fetching documents: {response.text}"
    except Exception as e:
        return f"Could not connect to FastAPI backend: {e}"


def upload_and_ingest_file(file, strategy):
    """Upload PDF file to FastAPI ingest endpoint."""
    if file is None:
        return "Please select a file to upload."
    try:
        filename = file.name
        with open(file.name, "rb") as f:
            files = {"file": (filename, f, "application/pdf")}
            data = {"chunking_strategy": strategy}
            response = httpx.post(f"{API_URL}/upload", files=files, data=data, timeout=30.0)

            if response.status_code == 202:
                res_data = response.json()
                return (
                    f"✅ Success: {res_data['message']}\n"
                    f"Strategy: {res_data['chunking_strategy']}\n"
                    "Please check the Ingested Documents table in a moment."
                )
            else:
                return f"❌ Ingestion failed: {response.text}"
    except Exception as e:
        return f"❌ Connection error: {e}"


def run_chunking_eval():
    """Call the evaluation runner script and return markdown output."""
    import subprocess
    try:
        result = subprocess.run(
            ["uv", "run", "python", "scripts/run_chunking_eval.py"],
            capture_output=True, text=True, check=True
        )
        return result.stdout
    except Exception as e:
        return f"Error running chunking evaluation script: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Streaming chat handler
# ─────────────────────────────────────────────────────────────────────────────

def stream_chat(query, history, strategy):
    """Generator: stream RAG answer tokens from /api/stream SSE endpoint.

    Yields incremental (query_input, history, source_inspector) tuples
    so Gradio updates the chatbot character-by-character.
    """
    if not query.strip():
        yield "", history, "Enter a question to inspect sources."
        return

    # ── Optimistically add user message so it appears immediately ──────────
    history = history + [{"role": "user", "content": query}]
    yield "", history, "⏳ *Thinking…*"

    payload = {
        "query": query,
        "chunking_strategy": strategy,
        "history": history[:-1],  # exclude the just-added user message
    }

    accumulated_answer = ""
    citations_data = []
    error_occurred = False

    try:
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", f"{API_URL}/stream", json=payload) as response:
                if response.status_code != 200:
                    error_msg = f"API Error ({response.status_code}): {response.text}"
                    history = history + [{"role": "assistant", "content": error_msg}]
                    yield "", history, "❌ Error fetching sources."
                    return

                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue

                    event_data = line[6:]  # strip "data: " prefix

                    # ── Terminal signals ────────────────────────────────────
                    if event_data == "[DONE]":
                        break

                    if event_data.startswith("[ERROR]"):
                        error_msg = event_data[7:].strip()
                        history = history + [{"role": "assistant", "content": f"❌ {error_msg}"}]
                        yield "", history, "❌ An error occurred."
                        error_occurred = True
                        break

                    if event_data.startswith("[CITATIONS]"):
                        # Parse citations JSON from the final metadata event
                        try:
                            citations_data = json.loads(event_data[11:].strip())
                        except json.JSONDecodeError:
                            citations_data = []
                        continue

                    # ── Token: append to accumulated answer ─────────────────
                    accumulated_answer += event_data

                    # Update chatbot with partial answer in real-time
                    partial_history = history + [{"role": "assistant", "content": accumulated_answer + " ▌"}]
                    yield "", partial_history, "⏳ *Generating…*"

    except Exception as e:
        error_msg = f"❌ Failed to connect to backend: {e}"
        history = history + [{"role": "assistant", "content": error_msg}]
        yield "", history, "❌ Connection error."
        return

    if error_occurred:
        return

    # ── Final update: remove cursor, build source inspector ────────────────
    final_history = history + [{"role": "assistant", "content": accumulated_answer}]

    inspector_md = "### Source Inspector\n\n"
    if not citations_data:
        inspector_md += "*No source documents retrieved or matched.*"
    else:
        for idx, c in enumerate(citations_data, 1):
            title = c.get("title") or "Unknown Document"
            section = c.get("section_path") or "root"
            idx_val = c.get("chunk_index", 0)
            text = c.get("text", "")
            inspector_md += (
                f"**{idx}. Source:** {title} > *{section}* (Chunk {idx_val})\n\n"
                f"```\n{text}\n```\n"
                f"---\n"
            )

    yield "", final_history, inspector_md


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

custom_css = """
body {
    background-color: #0f0f13 !important;
}
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    padding-top: 30px !important;
}
.feedback-header {
    text-align: center;
    margin-bottom: 20px;
}
.feedback-header h1 {
    color: #e0dbff;
    font-size: 2.5em;
    font-weight: 800;
    text-shadow: 0 0 10px rgba(138, 92, 246, 0.4);
}
/* Streaming cursor blink */
@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Gradio Blocks UI
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks() as demo:
    gr.HTML(
        '<div class="feedback-header">'
        '<h1>🤖 RAG Technical Chatbot</h1>'
        '<p style="color: #a3a3c2;">Docling Ingestion • OpenSearch Hybrid RAG • LangGraph • ⚡ Streaming</p>'
        '</div>'
    )

    with gr.Tabs():
        # ── TAB 1: CHAT & INSPECT ─────────────────────────────────────────
        with gr.TabItem("💬 Chat & Source Inspector"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Chat History",
                        height=500,
                    )

                    with gr.Row():
                        query_input = gr.Textbox(
                            show_label=False,
                            placeholder="Ask a question about the document corpus… (Shift+Enter for new line)",
                            scale=4,
                            lines=1,
                            max_lines=4,
                        )
                        submit_btn = gr.Button("⚡ Send", variant="primary", scale=1)

                    with gr.Row():
                        strategy_select = gr.Dropdown(
                            label="Retrieval Chunking Strategy",
                            choices=["semantic", "structural", "fixed_size", "recursive"],
                            value="semantic",
                            interactive=True,
                        )
                        clear_btn = gr.Button("🗑️ Clear Conversation")

                with gr.Column(scale=2):
                    source_inspector = gr.Markdown(
                        value="### Source Inspector\n\n*Retrieved chunks and citations will appear here after each query.*",
                        elem_id="inspector-pane",
                    )

            # ── Wire streaming events ──────────────────────────────────────
            # Both submit button and Enter key use the streaming generator.
            submit_btn.click(
                stream_chat,
                inputs=[query_input, chatbot, strategy_select],
                outputs=[query_input, chatbot, source_inspector],
            )
            query_input.submit(
                stream_chat,
                inputs=[query_input, chatbot, strategy_select],
                outputs=[query_input, chatbot, source_inspector],
            )
            clear_btn.click(
                lambda: ([], "### Source Inspector\n\n*Start a conversation to see source citations here.*"),
                outputs=[chatbot, source_inspector],
            )

        # ── TAB 2: ADMIN & EVALUATION ─────────────────────────────────────
        with gr.TabItem("⚙️ Ingestion & Chunking Evaluation"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Document Ingestion")
                    file_input = gr.File(label="Upload Technical PDF", file_types=[".pdf"])
                    ingest_strategy = gr.Dropdown(
                        label="Ingestion Chunking Strategy",
                        choices=["semantic", "structural", "fixed_size", "recursive"],
                        value="semantic",
                    )
                    ingest_btn = gr.Button("📥 Trigger Document Ingestion", variant="primary")
                    ingest_status = gr.Textbox(label="Ingestion Pipeline Status", interactive=False)

                    ingest_btn.click(
                        upload_and_ingest_file,
                        inputs=[file_input, ingest_strategy],
                        outputs=[ingest_status],
                    )

                with gr.Column():
                    gr.Markdown("### Ingested Documents Store")
                    refresh_btn = gr.Button("🔄 Refresh Documents List")
                    documents_table = gr.Markdown(value="*Click Refresh to list ingested documents.*")
                    refresh_btn.click(get_ingested_documents, outputs=[documents_table])

            gr.HTML("<hr style='border-color: #333;' />")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Chunking Evaluator Harness")
                    gr.HTML("<p style='color: #888;'>Run a local evaluation comparison of fixed-size, recursive markdown, custom structural, and semantic similarity chunkers on MRR and Hit Rate metrics.</p>")
                    eval_btn = gr.Button("📊 Run Comparison Evaluation", variant="secondary")

                with gr.Column(scale=3):
                    eval_results = gr.Markdown(
                        value="*Evaluation report will be generated here. (Note: parsing PDFs on CPU might take up to a minute)*"
                    )
                    eval_btn.click(run_chunking_eval, outputs=[eval_results])

    # Initial load
    demo.load(get_ingested_documents, outputs=[documents_table])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="purple", secondary_hue="slate"),
        css=custom_css,
    )
