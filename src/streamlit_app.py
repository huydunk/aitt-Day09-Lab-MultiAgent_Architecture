"""Streamlit UI for the Shopping Multi-Agent system.

Run:
    $env:PYTHONPATH="src"; streamlit run src/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shopping Assistant – Multi-Agent Flow",
    page_icon="🛒",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .node-box {
        border-radius: 10px;
        padding: 10px 18px;
        text-align: center;
        font-weight: 600;
        font-size: 14px;
        min-width: 130px;
        display: inline-block;
    }
    .node-active   { background:#d4edda; border:2px solid #28a745; color:#155724; }
    .node-inactive { background:#e9ecef; border:2px solid #adb5bd; color:#6c757d; }
    .node-response { background:#cce5ff; border:2px solid #004085; color:#004085; }
    .arrow { font-size:22px; color:#6c757d; padding:0 4px; line-height:40px; }
    .flow-row { display:flex; align-items:center; gap:6px; flex-wrap:wrap; margin:12px 0; }
    .tag-ok    { background:#d4edda; color:#155724; border-radius:4px; padding:2px 8px; font-size:12px; }
    .tag-warn  { background:#fff3cd; color:#856404; border-radius:4px; padding:2px 8px; font-size:12px; }
    .tag-err   { background:#f8d7da; color:#721c24; border-radius:4px; padding:2px 8px; font-size:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── helper: load assistant into session_state (survives errors gracefully) ────
def load_assistant():
    if "assistant" not in st.session_state:
        try:
            from app.graph import ShoppingAssistant
            with st.spinner("Loading model & index…"):
                st.session_state["assistant"] = ShoppingAssistant()
            st.session_state["assistant_error"] = None
        except Exception as e:
            st.session_state["assistant"] = None
            st.session_state["assistant_error"] = e
    return st.session_state["assistant"], st.session_state.get("assistant_error")


# ── helper: render flow diagram ───────────────────────────────────────────────
def render_flow(trace: list[dict]) -> None:
    visited = {t["node"] for t in trace}

    def node_html(label: str, node_key: str, style: str = "node-active") -> str:
        cls = style if node_key in visited else "node-inactive"
        return f'<div class="node-box {cls}">{label}</div>'

    arrow = '<span class="arrow">→</span>'
    fork_open  = '<div style="display:flex;flex-direction:column;gap:6px;">'
    fork_close = '</div>'

    html = (
        '<div class="flow-row">'
        + node_html("👤 User", "_user", "node-active")
        + arrow
        + node_html("🧭 Supervisor", "supervisor")
        + arrow
        + fork_open
        + node_html("📄 Policy Worker", "worker_policy")
        + node_html("🗄️ Data Worker", "worker_data")
        + fork_close
        + arrow
        + node_html("💬 Response Worker", "worker_response", "node-response")
        + arrow
        + node_html("✅ Answer", "_answer", "node-response")
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


# ── helper: render each node's details ───────────────────────────────────────
def render_supervisor(entry: dict) -> None:
    out = entry.get("output", {})
    status = out.get("status", "ok")
    tag = "tag-ok" if status == "ok" else "tag-warn"
    st.markdown(f'<span class="{tag}">{status}</span>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.metric("needs_policy", "✅ Yes" if out.get("needs_policy") else "❌ No")
    col2.metric("needs_data",   "✅ Yes" if out.get("needs_data")   else "❌ No")

    if out.get("clarification_question"):
        st.info(f"Clarification: {out['clarification_question']}")

    with st.expander("Raw JSON"):
        st.json(out)


def render_policy_worker(entry: dict) -> None:
    out = entry.get("output", {})
    hits = entry.get("hits", [])

    st.markdown("**RAG Hits retrieved from Chroma:**")
    for i, h in enumerate(hits, 1):
        score = round(1 - h.get("distance", 0), 3)
        with st.expander(f"Hit {i} — `{h['citation']}` (score {score})"):
            st.text(h["content"][:600] + ("…" if len(h["content"]) > 600 else ""))

    st.markdown("---")
    st.markdown("**LLM Summary:**")
    st.write(out.get("summary", "—"))

    facts = out.get("facts", [])
    if facts:
        st.markdown("**Key Facts:**")
        for f in facts:
            st.markdown(f"- {f}")

    citations = out.get("citations", [])
    if citations:
        st.markdown("**Citations:**")
        for c in citations:
            st.markdown(f"- `{c}`")

    with st.expander("Raw JSON"):
        st.json(out)


def render_data_worker(entry: dict) -> None:
    out = entry.get("output", {})
    messages = entry.get("messages", [])

    tool_calls = [m for m in messages if m.get("type") == "ai" and m.get("tool_calls")]
    tool_results = [m for m in messages if m.get("type") == "tool"]

    if tool_calls:
        st.markdown("**Tool calls made:**")
        for msg in tool_calls:
            for tc in msg.get("tool_calls", []):
                name = tc.get("name", "?")
                args = tc.get("args", {})
                st.markdown(f"🔧 `{name}({', '.join(f'{k}={v!r}' for k, v in args.items())})`")

    if tool_results:
        st.markdown("**Tool results:**")
        for msg in tool_results:
            with st.expander(f"Result from `{msg.get('tool_name', '?')}`"):
                try:
                    parsed = json.loads(msg["content"]) if isinstance(msg["content"], str) else msg["content"]
                    st.json(parsed)
                except Exception:
                    st.text(str(msg.get("content", "")))

    st.markdown("---")
    st.markdown("**LLM Summary:**")
    st.write(out.get("summary", "—"))

    facts = out.get("facts", [])
    if facts:
        st.markdown("**Key Facts:**")
        for f in facts:
            st.markdown(f"- {f}")

    not_found = out.get("not_found_entities", [])
    if not_found:
        st.warning(f"Not found: {', '.join(not_found)}")

    with st.expander("Raw JSON"):
        st.json(out)


def render_response_worker(entry: dict) -> None:
    out = entry.get("output", "")
    if out.startswith("Status: error"):
        st.error(out)
    else:
        st.markdown(out)


# ── sidebar: config status + reinit ──────────────────────────────────────────
with st.sidebar:
    st.header("Config")
    try:
        from app.config import Settings
        s = Settings.load()
        st.success(f"Provider: `{s.provider}`")
        st.info(f"Model: `{s.model}`")
    except Exception as e:
        st.error(f"Config error: {e}")

    if st.button("Reinitialize assistant", use_container_width=True):
        for key in ["assistant", "assistant_error", "payload"]:
            st.session_state.pop(key, None)
        st.rerun()

    if "assistant_error" in st.session_state and st.session_state["assistant_error"]:
        st.error(f"Init error:\n{st.session_state['assistant_error']}")

# ── main UI ───────────────────────────────────────────────────────────────────
st.title("🛒 Shopping Assistant — Multi-Agent Flow")
st.caption("Ask a question and watch each agent's work step by step.")

# ── question form ─────────────────────────────────────────────────────────────
with st.form("question_form"):
    question = st.text_input(
        "Question",
        placeholder='e.g. "Đơn hàng 1971 có được hoàn trả không?"',
    )
    col_btn, col_rebuild = st.columns([3, 1])
    submitted = col_btn.form_submit_button("▶ Run", use_container_width=True)
    rebuild = col_rebuild.form_submit_button("🔄 Rebuild Index", use_container_width=True)

# ── sample questions ──────────────────────────────────────────────────────────
with st.expander("Sample questions"):
    samples = [
        "Đơn hàng 1971 có được hoàn trả không?",
        "Chính sách hoàn trả hàng ra sao?",
        "Voucher của khách hàng C001 còn những mã nào dùng được?",
        "Giao hàng tiêu chuẩn thường mất bao lâu?",
        "Voucher của tôi còn dùng được không?",
        "Đơn hàng 9999 đang ở đâu?",
    ]
    for s in samples:
        st.markdown(f"- {s}")

# ── run ───────────────────────────────────────────────────────────────────────
if submitted and question.strip():
    assistant, err = load_assistant()
    if err:
        st.error(f"Failed to load assistant: {err}")
        st.info("Check your `.env` (LLM_MODEL, API key) then restart Streamlit.")
        st.stop()
    try:
        with st.spinner("Running agents…"):
            payload = assistant.ask(question.strip(), rebuild_index=False)
        st.session_state["payload"] = payload
    except Exception as e:
        st.error(f"Agent error: {e}")

if rebuild:
    assistant, err = load_assistant()
    if err:
        st.error(f"Failed to load assistant: {err}")
        st.stop()
    try:
        with st.spinner("Rebuilding Chroma index…"):
            assistant.vector_store.rebuild(assistant.settings.policy_path)
        st.cache_resource.clear()
        st.success("Index rebuilt. Cache cleared — reload the page.")
    except Exception as e:
        st.error(f"Rebuild error: {e}")

# ── display results ───────────────────────────────────────────────────────────
if "payload" in st.session_state:
    payload = st.session_state["payload"]
    trace: list[dict] = payload.get("trace", [])
    trace_by_node = {t["node"]: t for t in trace}

    st.markdown("---")
    st.subheader("Flow")
    render_flow(trace)

    st.markdown("---")
    st.subheader("Node Details")

    tabs_labels = ["🧭 Supervisor"]
    if "worker_policy" in trace_by_node:
        tabs_labels.append("📄 Policy Worker")
    if "worker_data" in trace_by_node:
        tabs_labels.append("🗄️ Data Worker")
    tabs_labels.append("💬 Response Worker")

    tabs = st.tabs(tabs_labels)
    tab_idx = 0

    with tabs[tab_idx]:
        if "supervisor" in trace_by_node:
            render_supervisor(trace_by_node["supervisor"])
        else:
            st.info("Supervisor did not run.")
    tab_idx += 1

    if "worker_policy" in trace_by_node:
        with tabs[tab_idx]:
            render_policy_worker(trace_by_node["worker_policy"])
        tab_idx += 1

    if "worker_data" in trace_by_node:
        with tabs[tab_idx]:
            render_data_worker(trace_by_node["worker_data"])
        tab_idx += 1

    with tabs[tab_idx]:
        if "worker_response" in trace_by_node:
            render_response_worker(trace_by_node["worker_response"])
        else:
            st.info("Response worker did not run.")

    st.markdown("---")
    with st.expander("Full payload (debug)"):
        st.json(payload)
