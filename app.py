import json
import time
import tempfile
import os
import streamlit as st
import httpx
import pandas as pd
from chromadb import PersistentClient
from pathlib import Path
from settings import Settings

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="FinLens · Document Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

:root {
    --navy:      #080d1a;
    --navy-2:    #0d1526;
    --navy-3:    #111d35;
    --navy-4:    #162040;
    --gold:      #c9a84c;
    --gold-dim:  #8a6e2f;
    --gold-glow: #c9a84c22;
    --slate:     #8fa3bf;
    --slate-dim: #4a5f7a;
    --white:     #e8edf5;
    --red:       #e05c5c;
    --red-dim:   #3d1616;
    --green:     #4caf82;
    --green-dim: #163d2b;
    --border:    #1e2f4a;
}
html, body, [class*="css"] { font-family:'JetBrains Mono',monospace; background-color:var(--navy) !important; color:var(--white); }
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding:0 2rem 2rem 2rem !important; max-width:100% !important; }
.topbar { background:linear-gradient(90deg,var(--navy-2) 0%,var(--navy-3) 60%,var(--navy-2) 100%); border-bottom:1px solid var(--gold-dim); padding:1rem 2rem; margin:0 -2rem 2.5rem -2rem; display:flex; align-items:center; justify-content:space-between; }
.topbar-logo { font-family:'Playfair Display',serif; font-size:1.5rem; font-weight:700; color:var(--gold); letter-spacing:0.02em; }
.topbar-sub  { font-size:0.62rem; color:var(--slate-dim); letter-spacing:0.15em; text-transform:uppercase; margin-left:0.75rem; }
.topbar-right { display:flex; gap:1rem; align-items:center; }
.topbar-tag  { font-size:0.58rem; color:var(--gold-dim); letter-spacing:0.15em; text-transform:uppercase; border:1px solid var(--gold-dim); padding:2px 8px; border-radius:1px; }
.sec-label { font-size:0.58rem; letter-spacing:0.22em; text-transform:uppercase; color:var(--gold); border-left:2px solid var(--gold); padding-left:0.6rem; margin-bottom:0.9rem; display:block; }
.metrics-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:0.6rem; margin-bottom:1rem; }
.metric { background:var(--navy-3); border:1px solid var(--border); border-radius:2px; padding:0.9rem 1.1rem; }
.metric-label { font-size:0.52rem; letter-spacing:0.18em; text-transform:uppercase; color:var(--slate-dim); margin-bottom:0.4rem; }
.metric-value { font-size:1.3rem; font-weight:600; color:var(--white); line-height:1; }
.metric-value.gold { color:var(--gold); }
.metric-value.small { font-size:0.72rem; color:var(--slate); word-break:break-all; line-height:1.4; }
.answer-box { background:var(--navy-3); border:1px solid var(--border); border-top:1px solid var(--gold-dim); border-radius:2px; padding:1.1rem 1.4rem; margin-bottom:0.9rem; }
.answer-label { font-size:0.52rem; letter-spacing:0.18em; text-transform:uppercase; color:var(--gold-dim); margin-bottom:0.6rem; }
.answer-text  { font-size:0.9rem; line-height:1.8; color:var(--white); font-weight:300; }
.anomaly-alert { background:var(--red-dim); border:1px solid var(--red); border-radius:2px; padding:0.9rem 1.4rem; margin-bottom:0.9rem; }
.anomaly-clear { background:var(--green-dim); border:1px solid var(--green); border-radius:2px; padding:0.9rem 1.4rem; margin-bottom:0.9rem; }
.anomaly-badge-red   { font-size:0.62rem; letter-spacing:0.12em; text-transform:uppercase; color:var(--red); font-weight:600; }
.anomaly-badge-green { font-size:0.62rem; letter-spacing:0.12em; text-transform:uppercase; color:var(--green); font-weight:600; }
.anomaly-reason { font-size:0.82rem; color:var(--red); margin-top:0.4rem; font-weight:300; }
.chunk-meta { font-size:0.58rem; color:var(--gold-dim); letter-spacing:0.1em; margin-bottom:0.3rem; }
.chunk-box  { background:var(--navy-4); border:1px solid var(--border); border-left:2px solid var(--gold-dim); border-radius:2px; padding:0.7rem 1rem; margin-bottom:0.5rem; font-size:0.68rem; line-height:1.65; color:var(--slate); white-space:pre-wrap; word-break:break-word; }
.empty-state { border:1px dashed var(--border); border-radius:2px; padding:4rem 2rem; text-align:center; }
.empty-icon  { font-size:2rem; color:var(--gold-dim); margin-bottom:1rem; }
.empty-text  { font-size:0.65rem; letter-spacing:0.12em; text-transform:uppercase; color:var(--slate-dim); line-height:2.2; }
.stButton > button { background:transparent !important; border:1px solid var(--gold-dim) !important; color:var(--gold) !important; border-radius:2px !important; font-family:'JetBrains Mono',monospace !important; font-size:0.72rem !important; letter-spacing:0.1em !important; text-transform:uppercase !important; padding:0.5rem 1.5rem !important; width:100% !important; }
.stButton > button:hover { background:var(--gold-glow) !important; border-color:var(--gold) !important; }
.stTextArea > div > div > textarea { background:var(--navy-3) !important; border:1px solid var(--border) !important; border-radius:2px !important; color:var(--white) !important; font-family:'JetBrains Mono',monospace !important; font-size:0.8rem !important; caret-color:var(--gold) !important; }
.stTextArea > div > div > textarea:focus { border-color:var(--gold-dim) !important; box-shadow:none !important; }
.stSelectbox > div > div { background:var(--navy-3) !important; border:1px solid var(--border) !important; color:var(--white) !important; font-family:'JetBrains Mono',monospace !important; font-size:0.78rem !important; border-radius:2px !important; }
.stFileUploader { background:var(--navy-3) !important; border:1px dashed var(--border) !important; border-radius:2px !important; }
.stExpander { background:var(--navy-2) !important; border:1px solid var(--border) !important; border-radius:2px !important; }
.stSuccess,.stError,.stInfo,.stWarning { font-family:'JetBrains Mono',monospace !important; font-size:0.75rem !important; border-radius:2px !important; }
label { font-family:'JetBrains Mono',monospace !important; font-size:0.72rem !important; color:var(--slate) !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="topbar">
    <div style="display:flex;align-items:baseline;">
        <span class="topbar-logo">◈ FinLens</span>
        <span class="topbar-sub">Financial Document Intelligence Platform</span>
    </div>
    <div class="topbar-right">
        <span class="topbar-tag">RAG Pipeline</span>
        <span class="topbar-tag">Anomaly Detection</span>
        <span class="topbar-tag">BFSI · Agentic AI</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=10)
def get_ingested_documents():
    try:
        if Settings.VECTOR_DB == "qdrant":
            from qdrant_client import QdrantClient

            client = QdrantClient(
                url=Settings.QDRANT_URL, api_key=Settings.QDRANT_API_KEY
            )
            results = client.scroll(
                collection_name=Settings.COLLECTION_NAME,
                limit=100,
                with_payload=True,
            )[0]
            docs = list(
                {r.payload.get("source") for r in results if r.payload.get("source")}
            )
            return sorted(docs)
        else:
            from chromadb import PersistentClient

            client = PersistentClient(path=Settings.CHROMA_PATH)
            collection = client.get_or_create_collection(Settings.COLLECTION_NAME)
            results = collection.get(include=["metadatas"])
            docs = list({m["source"] for m in results["metadatas"] if m.get("source")})
            return sorted(docs)
    except Exception:
        return []


def get_chunks_for_doc(doc_name: str):
    try:
        if Settings.VECTOR_DB == "qdrant":
            from qdrant_client import QdrantClient
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            client = QdrantClient(
                url=Settings.QDRANT_URL, api_key=Settings.QDRANT_API_KEY
            )
            results = client.scroll(
                collection_name=Settings.COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="source", match=MatchValue(value=doc_name))
                    ]
                ),
                limit=50,
                with_payload=True,
            )[0]
            return [
                (
                    r.payload.get("text", ""),
                    {
                        "source": r.payload.get("source", ""),
                        "chunk_index": r.payload.get("chunk_index", 0),
                    },
                )
                for r in results
            ]
        else:
            from chromadb import PersistentClient

            client = PersistentClient(path=Settings.CHROMA_PATH)
            collection = client.get_or_create_collection(Settings.COLLECTION_NAME)
            results = collection.get(
                where={"source": doc_name}, include=["documents", "metadatas"]
            )
            return list(zip(results["documents"], results["metadatas"]))
    except Exception:
        return []


# ── Pipeline renderer (fully inlined styles) ──────────────────────────────────
def render_pipeline(steps: dict, total_ms: int = None) -> str:
    STATE = {
        "waiting": {
            "box": "background:#111d35;border:1px solid #1e2f4a;border-radius:4px;padding:14px 8px;text-align:center;opacity:0.5;flex:1;min-width:0;",
            "icon": "font-size:20px;margin-bottom:6px;display:block;color:#4a5f7a;",
            "name": "font-size:0.6rem;font-weight:500;color:#4a5f7a;margin:0 0 5px;line-height:1.3;letter-spacing:0.05em;font-family:'JetBrains Mono',monospace;",
            "time": "font-size:0.58rem;color:#4a5f7a;letter-spacing:0.08em;font-family:'JetBrains Mono',monospace;",
            "icon_char": "⏱",
        },
        "running": {
            "box": "background:#1f3a5f;border:1px solid #58a6ff;border-radius:4px;padding:14px 8px;text-align:center;flex:1;min-width:0;",
            "icon": "font-size:20px;margin-bottom:6px;display:block;color:#58a6ff;",
            "name": "font-size:0.6rem;font-weight:500;color:#58a6ff;margin:0 0 5px;line-height:1.3;letter-spacing:0.05em;font-family:'JetBrains Mono',monospace;",
            "time": "font-size:0.58rem;color:#58a6ff;letter-spacing:0.08em;font-family:'JetBrains Mono',monospace;",
            "icon_char": "⟳",
        },
        "done": {
            "box": "background:#163d2b55;border:1px solid #4caf82;border-radius:4px;padding:14px 8px;text-align:center;flex:1;min-width:0;",
            "icon": "font-size:20px;margin-bottom:6px;display:block;color:#4caf82;",
            "name": "font-size:0.6rem;font-weight:500;color:#4caf82;margin:0 0 5px;line-height:1.3;letter-spacing:0.05em;font-family:'JetBrains Mono',monospace;",
            "time": "font-size:0.58rem;color:#4caf82;font-weight:600;letter-spacing:0.08em;font-family:'JetBrains Mono',monospace;",
            "icon_char": "✓",
        },
        "error": {
            "box": "background:#3d1616;border:1px solid #e05c5c;border-radius:4px;padding:14px 8px;text-align:center;flex:1;min-width:0;",
            "icon": "font-size:20px;margin-bottom:6px;display:block;color:#e05c5c;",
            "name": "font-size:0.6rem;font-weight:500;color:#e05c5c;margin:0 0 5px;line-height:1.3;letter-spacing:0.05em;font-family:'JetBrains Mono',monospace;",
            "time": "font-size:0.58rem;color:#e05c5c;letter-spacing:0.08em;font-family:'JetBrains Mono',monospace;",
            "icon_char": "✗",
        },
    }

    boxes = ""
    for name, info in steps.items():
        state = info.get("state", "waiting")
        elapsed = info.get("elapsed_ms")
        s = STATE.get(state, STATE["waiting"])
        if state == "done" and elapsed is not None:
            t = f"{elapsed/1000:.2f}s"
        elif state == "running":
            t = "running..."
        elif state == "error":
            t = "error"
        else:
            t = "waiting"
        boxes += f'<div style="{s["box"]}"><span style="{s["icon"]}">{s["icon_char"]}</span><div style="{s["name"]}">{name}</div><div style="{s["time"]}">{t}</div></div>'

    if total_ms is not None:
        total_bar = f"<div style=\"margin-top:8px;background:#111d35;border:1px solid #8a6e2f;border-radius:4px;padding:8px 14px;font-size:0.62rem;color:#c9a84c;letter-spacing:0.12em;font-family:'JetBrains Mono',monospace;\">◈ Total pipeline time: {total_ms/1000:.2f}s</div>"
    else:
        total_bar = "<div style=\"margin-top:8px;background:#111d35;border:1px solid #1e2f4a;border-radius:4px;padding:8px 14px;font-size:0.62rem;color:#4a5f7a;letter-spacing:0.12em;font-family:'JetBrains Mono',monospace;\">◈ Pipeline running...</div>"

    return (
        f'<div style="display:flex;gap:8px;margin-bottom:4px;">{boxes}</div>{total_bar}'
    )


def score_color(s):
    if s >= 0.75:
        return "#4caf82"
    if s >= 0.5:
        return "#c9a84c"
    return "#e05c5c"


# ── Layout ────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2], gap="large")

# ════════════════════════════════════
# LEFT PANEL
# ════════════════════════════════════
with col_left:

    st.markdown(
        '<span class="sec-label">01 · Document Library</span>', unsafe_allow_html=True
    )
    docs = get_ingested_documents()

    if docs:
        selected_doc = st.selectbox(
            "Select document", options=docs, label_visibility="collapsed"
        )
        st.session_state["selected_doc"] = selected_doc
        with st.expander(f"◈ View chunks — {selected_doc}", expanded=False):
            chunks = get_chunks_for_doc(selected_doc)
            if chunks:
                for doc_text, meta in chunks:
                    st.markdown(
                        f'<div class="chunk-meta">CHUNK {meta.get("chunk_index","?")} · {meta.get("source","")}</div><div class="chunk-box">{doc_text}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div style="font-size:0.72rem;color:#4a5f7a;">No chunks found.</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            '<div style="border:1px dashed #1e2f4a;padding:1.2rem;border-radius:2px;font-size:0.68rem;color:#4a5f7a;text-align:center;letter-spacing:0.1em;">NO DOCUMENTS INGESTED YET</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

    # 02 · Ingest
    st.markdown(
        '<span class="sec-label">02 · Ingest New Document</span>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload PDF", type=["pdf"], label_visibility="collapsed"
    )

    if uploaded_file:
        st.markdown(
            f'<div style="font-size:0.68rem;color:#c9a84c;margin:0.4rem 0 0.8rem 0;">◈ {uploaded_file.name} · {uploaded_file.size/1024:.1f} KB</div>',
            unsafe_allow_html=True,
        )

    ingest_clicked = st.button(
        "⊕  Ingest Document", disabled=uploaded_file is None, key="ingest_btn"
    )

    if ingest_clicked and uploaded_file:
        st.session_state["ingest_file_name"] = uploaded_file.name
        st.session_state["ingest_file_content"] = uploaded_file.getvalue()
        st.session_state["show_ingest_pipeline"] = True

    if st.session_state.get("show_ingest_pipeline"):
        st.markdown(
            '<span class="sec-label" style="margin-top:1rem;">Ingest Pipeline</span>',
            unsafe_allow_html=True,
        )
        store_label = (
            "Storing in Qdrant"
            if Settings.VECTOR_DB == "qdrant"
            else "Storing in ChromaDB"
        )
        store_label = (
            "Storing in Qdrant"
            if Settings.VECTOR_DB == "qdrant"
            else "Storing in ChromaDB"
        )
        ingest_steps_def = ["PDF Extraction", "Text Chunking", "Embedding", store_label]
        ingest_state = {
            s: {"state": "waiting", "elapsed_ms": None} for s in ingest_steps_def
        }
        pipeline_ph = st.empty()
        pipeline_ph.markdown(
            render_pipeline(ingest_state), unsafe_allow_html=True
        )  # ← fixed

        try:
            fname = st.session_state["ingest_file_name"]
            fcontent = st.session_state["ingest_file_content"]
            total_ms = None

            with httpx.Client(timeout=300) as client:
                with client.stream(
                    "POST",
                    f"{API_BASE}/ingest/stream",
                    files={"file": (fname, fcontent, "application/pdf")},
                ) as r:
                    buffer = ""
                    for chunk in r.iter_text():
                        buffer += chunk
                        lines = buffer.split("\n")
                        buffer = lines.pop()
                        for line in lines:
                            line = line.strip()
                            if not line.startswith("data: "):
                                continue
                            try:
                                evt = json.loads(line[6:])
                                step = evt.get("step", "")
                                evtstatus = evt.get("status", "")
                                elapsed = evt.get("elapsed_ms")
                                if step == "complete":
                                    total_ms = elapsed
                                    pipeline_ph.markdown(
                                        render_pipeline(
                                            ingest_state, total_ms=total_ms
                                        ),
                                        unsafe_allow_html=True,
                                    )
                                elif step in ingest_state:
                                    if evtstatus == "running":
                                        ingest_state[step]["state"] = "running"
                                    elif evtstatus == "done":
                                        ingest_state[step]["state"] = "done"
                                        ingest_state[step]["elapsed_ms"] = elapsed
                                    elif evtstatus == "error":
                                        ingest_state[step]["state"] = "error"
                                    pipeline_ph.markdown(
                                        render_pipeline(
                                            ingest_state, total_ms=total_ms
                                        ),
                                        unsafe_allow_html=True,
                                    )
                            except Exception:
                                pass

            st.session_state["show_ingest_pipeline"] = False
            st.session_state["last_ingest_state"] = ingest_state
            st.session_state["last_ingest_total_ms"] = total_ms
            st.cache_data.clear()
            st.rerun()

        except httpx.ConnectError:
            st.error("Cannot connect to FastAPI on port 8000")
            st.session_state["show_ingest_pipeline"] = False
        except Exception as e:
            st.error(f"Ingest failed: {e}")
            st.session_state["show_ingest_pipeline"] = False

    if st.session_state.get("last_ingest_state") and not st.session_state.get(
        "show_ingest_pipeline"
    ):
        st.markdown(
            '<span class="sec-label" style="margin-top:1rem;">Last Ingest Pipeline</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            render_pipeline(
                st.session_state["last_ingest_state"],
                total_ms=st.session_state.get("last_ingest_total_ms"),
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

    # 03 · Query
    st.markdown('<span class="sec-label">03 · Query</span>', unsafe_allow_html=True)
    query = st.text_area(
        "Query",
        placeholder="What are the large withdrawals?\nAre there suspicious transactions?\nSummarise all credits this month.",
        height=130,
        label_visibility="collapsed",
        key="query_input",
    )
    st.markdown(
        '<div style="font-size:0.58rem;color:#4a5f7a;letter-spacing:0.1em;margin-top:0.25rem;">SHIFT+ENTER FOR NEW LINE</div>',
        unsafe_allow_html=True,
    )

    # Button always enabled — validated on click
    analyse_clicked = st.button("◈  Run Analysis", key="analyse_btn")


# ════════════════════════════════════
# RIGHT PANEL
# ════════════════════════════════════
with col_right:

    st.markdown(
        '<span class="sec-label">04 · Analysis Results</span>', unsafe_allow_html=True
    )

    if analyse_clicked:
        if not query.strip():
            st.warning("Please enter a query first.")
        elif not docs:
            st.warning("Please ingest a document first.")
        else:
            analyze_steps_def = [
                "Query Received",
                "Retrieving Chunks",
                "Reasoning (LLM)",
                "Validation",
                "RAG Evaluation",
            ]
            analyze_state = {
                s: {"state": "waiting", "elapsed_ms": None} for s in analyze_steps_def
            }

            pipeline_ph = st.empty()
            pipeline_ph.markdown(
                render_pipeline(analyze_state), unsafe_allow_html=True
            )  # ← fixed

            result = None
            total_ms = None

            try:
                with httpx.Client(timeout=300) as client:
                    with client.stream(
                        "POST", f"{API_BASE}/analyze/stream", json={"query": query}
                    ) as r:
                        buffer = ""
                        for chunk in r.iter_text():
                            buffer += chunk
                            lines = buffer.split("\n")
                            buffer = lines.pop()
                            for line in lines:
                                line = line.strip()
                                if not line.startswith("data: "):
                                    continue
                                try:
                                    evt = json.loads(line[6:])
                                    step = evt.get("step", "")
                                    evtstatus = evt.get("status", "")
                                    elapsed = evt.get("elapsed_ms")
                                    data = evt.get("data")
                                    if step == "complete":
                                        total_ms = elapsed
                                        if evtstatus == "done" and data:
                                            result = data
                                        pipeline_ph.markdown(
                                            render_pipeline(
                                                analyze_state, total_ms=total_ms
                                            ),
                                            unsafe_allow_html=True,
                                        )
                                    elif step in analyze_state:
                                        if evtstatus == "running":
                                            analyze_state[step]["state"] = "running"
                                        elif evtstatus == "done":
                                            analyze_state[step]["state"] = "done"
                                            analyze_state[step]["elapsed_ms"] = elapsed
                                        elif evtstatus == "error":
                                            analyze_state[step]["state"] = "error"
                                        pipeline_ph.markdown(
                                            render_pipeline(
                                                analyze_state, total_ms=total_ms
                                            ),
                                            unsafe_allow_html=True,
                                        )
                                except Exception:
                                    pass

                if result:
                    st.session_state["last_result"] = result
                else:
                    st.session_state.pop("last_result", None)

            except httpx.ConnectError:
                st.error("Cannot connect to FastAPI on port 8000")
            except httpx.ReadTimeout:
                st.error("LLM timed out — try again")
            except Exception as e:
                st.error(f"Error: {e}")

    # ── Results ───────────────────────────────────────────────────────────────
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        data = result.get("data", {})

        txn_count = len(data.get("transactions", []))
        total = data.get("total_amount")
        confidence = data.get("confidence", 0)
        source = data.get("source_document", "—")
        total_str = f"₹{total:,.0f}" if total is not None else "—"
        conf_str = f"{confidence * 100:.0f}%"

        st.markdown(
            f"""
        <div class="metrics-grid">
            <div class="metric"><div class="metric-label">Transactions</div><div class="metric-value">{txn_count}</div></div>
            <div class="metric"><div class="metric-label">Net Amount</div><div class="metric-value">{total_str}</div></div>
            <div class="metric"><div class="metric-label">Confidence</div><div class="metric-value gold">{conf_str}</div></div>
            <div class="metric"><div class="metric-label">Source</div><div class="metric-value small">{source}</div></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
        <div class="answer-box">
            <div class="answer-label">◈ Insight</div>
            <div class="answer-text">{data.get("answer", "No answer returned.")}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if data.get("anomaly_flag"):
            st.markdown(
                f"""
            <div class="anomaly-alert">
                <div class="anomaly-badge-red">⚠ Anomaly Detected</div>
                <div class="anomaly-reason">{data.get("anomaly_reason", "")}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
            <div class="anomaly-clear">
                <div class="anomaly-badge-green">✓ No Anomalies Detected</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        transactions = data.get("transactions", [])
        if transactions:
            st.markdown(
                '<span class="sec-label" style="margin-top:0.5rem;">Transaction Breakdown</span>',
                unsafe_allow_html=True,
            )
            df = pd.DataFrame(transactions)
            if "amount" in df.columns:
                df["amount"] = df["amount"].apply(
                    lambda x: f"₹{x:,.0f}" if x >= 0 else f"-₹{abs(x):,.0f}"
                )
            df.columns = [c.replace("_", " ").title() for c in df.columns]
            st.dataframe(df, use_container_width=True, hide_index=True)

        evaluation = result.get("evaluation")
        if evaluation:
            st.markdown(
                '<span class="sec-label" style="margin-top:0.5rem;">05 · RAG Evaluation</span>',
                unsafe_allow_html=True,
            )
            ctx = evaluation.get("context_precision", 0)
            faith = evaluation.get("faithfulness", 0)
            ans = evaluation.get("answer_relevancy", 0)
            evaluator = evaluation.get("evaluator", "—")
            eval_ms = evaluation.get("latency_ms", 0)
            st.markdown(
                f"""
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.6rem;margin-bottom:0.6rem;">
                <div class="metric"><div class="metric-label">Context Precision</div><div class="metric-value" style="color:{score_color(ctx)}">{ctx:.2f}</div></div>
                <div class="metric"><div class="metric-label">Faithfulness</div><div class="metric-value" style="color:{score_color(faith)}">{faith:.2f}</div></div>
                <div class="metric"><div class="metric-label">Answer Relevancy</div><div class="metric-value" style="color:{score_color(ans)}">{ans:.2f}</div></div>
            </div>
            <div style="font-size:0.58rem;color:#4a5f7a;letter-spacing:0.1em;">Evaluator: {evaluator} · Latency: {eval_ms}ms</div>
            """,
                unsafe_allow_html=True,
            )

        with st.expander("◈ Raw JSON response", expanded=False):
            st.json(result)

    else:
        if not analyse_clicked:
            st.markdown(
                """
            <div class="empty-state">
                <div class="empty-icon">◈</div>
                <div class="empty-text">
                    Select a document from the library<br>
                    enter your query<br>
                    and run analysis
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div style="margin-top:3rem;padding-top:1rem;border-top:1px solid #1e2f4a;
            font-size:0.55rem;color:#2a3f5a;text-align:center;letter-spacing:0.18em;text-transform:uppercase;">
    FinLens · Document Intelligence System · Project B · Agentic AI Portfolio
</div>
""",
    unsafe_allow_html=True,
)
