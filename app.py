"""Streamlit UI for the AI Newsletter Agent.

Run with:  streamlit run app.py
Everything the LLM produces is validated against the Pydantic schemas; the app
only ever displays validated data (``AgentResult``) plus the run state.
"""

from __future__ import annotations

import json
import sys
import threading
import time as _time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from config.settings import configure_logging, get_settings
from schemas import AgentMode

# Model persistence config
MODEL_CONFIG_PATH = Path(__file__).resolve().parent / "outputs" / "model_config.json"

def load_model_config() -> dict:
    """Load the model configuration from the outputs folder."""
    if MODEL_CONFIG_PATH.exists():
        try:
            with open(MODEL_CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_model_config(config: dict) -> None:
    """Save the model configuration to the outputs folder."""
    MODEL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

st.set_page_config(page_title="AI Newsletter Agent", page_icon="📰", layout="wide", initial_sidebar_state="expanded")
configure_logging("WARNING")

# Show full tracebacks in the browser instead of a blank page.
st.set_option("client.showErrorDetails", True)

# Force dark theme
st.markdown("""
    <style>
    :root {
        --primary-color: #1f77b4;
        --background-color: #0e1117;
        --secondary-background-color: #262730;
        --text-color: #fafafa;
    }
    .stApp {
        background-color: #0e1117;
    }
    </style>
    """, unsafe_allow_html=True)

try:
    from progress import (
        NODE_ORDER,
        SPAN_ORDER,
        RunProgress,
        make_callback,
    )

    SPAN_ORDER = list(SPAN_ORDER)  # ensure list
except Exception as _progress_exc:  # pragma: no cover - defensive
    st.error(f"Could not load progress.py: {_progress_exc}")
    st.code(traceback.format_exc())
    st.stop()

settings = get_settings(refresh=True)

# Per-run UI state. The heavy run lives in a background thread so the
# browser keeps responding; results arrive through a queue.
if "run_progress" not in st.session_state:
    st.session_state.run_progress = RunProgress()
if "run_log" not in st.session_state:
    st.session_state.run_log = []
if "run_thread" not in st.session_state:
    st.session_state.run_thread = None
if "run_result" not in st.session_state:
    st.session_state.run_result = None
if "run_error" not in st.session_state:
    st.session_state.run_error = None
if "generating" not in st.session_state:
    st.session_state.generating = False
if "last_run" not in st.session_state:
    st.session_state.last_run = None


# Sidebar inputs before any rerun-sensitive logic.
_MODES: list[AgentMode] = [AgentMode.FULLY_AUTONOMOUS, AgentMode.HUMAN_IN_LOOP]


def _modes() -> list[AgentMode]:  # backward-compat: older revisions called _modes()
    return list(_MODES)


with st.sidebar:
    st.header("Newsletter goal")
    goal = st.text_area(
        "What should the newsletter cover?",
        height=140,
        placeholder="e.g. This week in AI, ML infra, and applied research.",
        key="goal",
    )
    st.divider()
    st.header("Run settings")
    mode = st.selectbox(
        "Mode",
        options=list(_MODES),
        index=0,
        format_func=lambda m: getattr(m, "label", str(m)),
    )
    st.divider()
    
    # Load saved model config for smart defaults
    model_config = load_model_config()
    saved_model = model_config.get("last_working_model", "")
    
    # Use Groq only - use saved model as default if available
    default_model = saved_model if saved_model else settings.groq_model
    model = st.text_input("Model (Groq slug)", value=default_model)
    
    target = st.number_input("Target articles", min_value=1, max_value=6, value=min(settings.target_articles, 4))
    window = st.number_input("Recency window (days)", min_value=0, max_value=60, value=min(settings.recency_window_days, 7))
    
    # Save intermediate results option
    save_intermediate = st.checkbox(
        "Save intermediate results",
        value=False,
        help="Save all artifacts (run.json, articles.xlsx, summaries.json) except newsletter.html"
    )

    st.divider()
    _has_key = settings.has_groq
    run_clicked = st.button(
        "Generate newsletter",
        type="primary",
        use_container_width=True,
        disabled=(not _has_key or st.session_state.generating),
    )
    if not _has_key:
        st.warning("GROQ_API_KEY not configured; button disabled.")


try:
    from tools.article_extractor import extraction_backends

    backends = extraction_backends()
except Exception:  # pragma: no cover - defensive
    backends = {}


def _node_label(node: str) -> str:
    return {
        "parse_goal": "Goal",
        "plan_research": "Planning",
        "research": "Research",
        "evaluate": "Evaluation",
        "select": "Selection",
        "summarize": "Summarization",
        "generate": "Generation",
        "critique": "Critique",
        "human_review": "Review",
        "save_output": "Save",
    }.get(node, node.replace("_", " ").title())


def _render_node_stepper(progress) -> None:
    """Vertical 'nodes of lights' stepper — one row per graph node.

    Pending nodes are grey; completed nodes glow green; the active node
    pulses orange; failures show a red ✕.
    """
    for node in NODE_ORDER:
        label = _node_label(node)
        if node in progress.failed_nodes:
            dot, color, extra = "✕", "#e74c3c", "text-shadow: 0 0 8px #e74c3c"
        elif node in progress.completed_nodes:
            dot, color = "●", "#27ae60"
            extra = "text-shadow: 0 0 12px #27ae60, 0 0 20px #27ae60"
        elif node == progress.last_node:
            dot, color = "●", "#f39c12"
            extra = "text-shadow: 0 0 8px #f39c12"
        else:
            dot, color, extra = "○", "#95a5a6", ""
        st.markdown(
            f"<div style='margin:2px 0; padding:2px'><span style='color:{color};font-size:1.2em;{extra}'>{dot}</span> "
            f"<small style='color:{color};vertical-align:top;margin-left:6px'>{label}</small></div>",
            unsafe_allow_html=True,
        )


def _render_progress_panel() -> None:
    """Full progress panel rendered into a container (used during live polling)."""
    progress = st.session_state.run_progress
    with st.container():
        st.markdown("### 📋 Newsletter Progress")
        _render_node_stepper(progress)
        st.markdown("---")
        
        # Visual progress bar with percentage
        total = len(NODE_ORDER)
        idx = len(progress.completed_nodes)
        frac = idx / total if total else 0.0
        pct = int(frac * 100)
        
        # Custom progress bar with color
        bar_color = "#27ae60" if pct == 100 else "#f39c12" if pct > 0 else "#3498db"
        st.markdown(f"""
        <div style='margin: 10px 0;'>
            <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                <strong>Overall Progress: {pct}%</strong>
                <span>Step {idx} of {total}</span>
            </div>
            <div style='background: #262730; border-radius: 10px; height: 24px; overflow: hidden;'>
                <div style='width: {pct}%; background: linear-gradient(90deg, {bar_color}, {bar_color}aa); 
                     height: 100%; transition: width 0.3s ease; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;'>
                    {'✅ Complete!' if pct == 100 else f'{pct}%'}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Current status
        elapsed = progress.elapsed
        node_info = {
            "parse_goal": "🎯 Parsing your goal...",
            "plan_research": "📋 Planning research queries...",
            "research": "🔎 Fetching articles from sources...",
            "evaluate": "📊 Scoring articles for relevance...",
            "select": "✅ Selecting best articles...",
            "summarize": "✍️ Writing summaries...",
            "generate": "📰 Generating newsletter...",
            "critique": "🔍 Reviewing quality...",
            "human_review": "👤 Awaiting approval...",
            "save_output": "💾 Saving artifacts...",
        }
        current_msg = node_info.get(progress.last_node, f"Working on {progress.last_node}...")
        st.info(f"⏱️ {elapsed:.1f}s elapsed — {current_msg}")
        
        # Live metrics row
        if progress.raw_article_count or progress.article_count:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📥 Raw", progress.raw_article_count)
            col2.metric("🔍 Deduped", progress.article_count)
            if progress.raw_article_count > 0:
                col3.metric("📊 Retention", f"{progress.article_count / progress.raw_article_count * 100:.0f}%")
            if progress.input_tokens or progress.output_tokens:
                col4.metric("🔤 Tokens", f"{progress.input_tokens + progress.output_tokens:,}")


def _run(*, on_progress=None):
    from agent.graph import run_newsletter_state

    # Speed profile: fewer articles, tighter timeouts, fewer sources per run.
    # REDUCED CALLS: minimal sources, no revisions, smaller batches
    overrides = {
        "target_articles": min(int(target), 4),  # Reduced from 6
        "max_articles_to_collect": 10,  # Reduced from 18
        "recency_window_days": int(window),
        "max_revisions": 0,  # DISABLED - no revision loop saves many LLM calls
        "research_max_sources": 3,  # Reduced from 4
        "max_items_per_source": 5,  # Reduced from 8
        "enrich_article_limit": 4,  # Reduced from 6
        "request_timeout_seconds": 30.0,
        "research_max_workers": 4,
        "llm_timeout_seconds": 120.0,
        "save_intermediate": save_intermediate,
    }
    return run_newsletter_state(
        goal.strip(),
        mode=mode,
        model=(model.strip() or None),
        overrides=overrides,
        on_progress=on_progress,
    )


if st.session_state.run_thread is not None:
    thread = st.session_state.run_thread
    progress = st.session_state.run_progress
    if thread.is_alive():
        _render_progress_panel()
        st.caption(f"Running… {progress.elapsed:.1f}s elapsed")
        _time.sleep(0.5)
        st.rerun()
    else:
        thread.join(timeout=1)
        if st.session_state.run_error is not None:
            st.session_state.generating = False
            st.session_state.run_thread = None
            st.error(f"The run failed: {st.session_state.run_error}")
            st.code(traceback.format_exc())
            st.stop()
        if st.session_state.run_result is not None:
            st.session_state["last_run"] = st.session_state.run_result
        st.session_state.generating = False
        st.session_state.run_thread = None
        st.session_state.run_result = None
        st.session_state.run_error = None
        st.rerun()
    st.stop()

if run_clicked:
    if not str(st.session_state.get("goal") or goal or "").strip():
        st.warning("Type a topic first, then click Generate.")
        st.stop()
    if st.session_state.generating:
        st.info("A run is already in progress — please wait.")
        st.stop()
    _has_key = settings.has_groq
    if not _has_key:
        st.error("GROQ_API_KEY not configured. Add it to newsletter-agent/.env and restart Streamlit.")
        st.stop()
    st.session_state.generating = True
    st.session_state.run_cancelled = False
    st.session_state.run_error = None
    st.session_state.run_result = None

    progress = RunProgress()
    callback = make_callback(progress=progress)
    st.session_state.run_progress = progress

    def _do_run():
        try:
            result = _run(on_progress=callback)
            st.session_state.run_result = result
        except Exception as exc:
            st.session_state.run_error = exc

    thread = threading.Thread(target=_do_run, daemon=True)
    st.session_state.run_thread = thread
    thread.start()

    _render_node_stepper(progress)
    st.caption("🚀 Generating newsletter…")
    st.rerun()

final = st.session_state.get("last_run")
if isinstance(final, dict) and final:
    progress = st.session_state.run_progress
    result = final.get("final_output")
    if result is not None and result.success:
        # Save the working model for next run (smart switch)
        if model.strip():
            save_model_config({"last_working_model": model.strip()})
        
        # Trigger success notification on first render after success
        if not st.session_state.get("success_shown", False):
            st.session_state.success_shown = True
            st.balloons()
        
        # ============ BEAUTIFUL COMPLETION UI ============
        st.markdown("---")
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1e3a2e 0%, #27ae60 100%); 
             padding: 2rem; border-radius: 16px; margin: 1rem 0; text-align: center;'>
            <h1 style='color: white; margin: 0; font-size: 2rem;'>🎉 Newsletter Ready!</h1>
            <p style='color: #e8f5e9; margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
                Your newsletter has been generated successfully
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Stats cards
        stat_cols = st.columns(5)
        stat_cols[0].metric("📰 Sections", len(result.summaries))
        stat_cols[1].metric("📥 Raw Articles", len(result.research_articles))
        stat_cols[2].metric("✅ Selected", len(result.selected_articles))
        stat_cols[3].metric("⏱️ Time", f"{progress.elapsed:.1f}s")
        stat_cols[4].metric("🔤 Tokens", f"{(progress.input_tokens + progress.output_tokens):,}")
        
        st.markdown("### 📥 Download Your Newsletter")
        
        # Large, prominent download buttons
        dl_col1, dl_col2 = st.columns(2)
        
        if result.html_content:
            with dl_col1:
                st.markdown("""
                <div style='background: #1e3a5f; padding: 1.5rem; border-radius: 12px; 
                     text-align: center; border: 2px solid #3498db;'>
                    <h3 style='color: #3498db; margin: 0 0 0.5rem 0;'>🌐 HTML Format</h3>
                    <p style='color: #aed6f1; margin: 0 0 1rem 0; font-size: 0.9rem;'>
                        Best for email clients & web viewing
                    </p>
                </div>
                """, unsafe_allow_html=True)
                st.download_button(
                    "📥 DOWNLOAD HTML",
                    data=result.html_content,
                    file_name=Path(result.html_path).name if result.html_path else "newsletter.html",
                    mime="text/html",
                    use_container_width=True,
                    key="dl_html_main",
                    type="primary",
                )
        
        if result.markdown_content:
            with dl_col2:
                st.markdown("""
                <div style='background: #3a1e3f; padding: 1.5rem; border-radius: 12px; 
                     text-align: center; border: 2px solid #e74c3c;'>
                    <h3 style='color: #e74c3c; margin: 0 0 0.5rem 0;'>📝 Markdown Format</h3>
                    <p style='color: #f5b7b1; margin: 0 0 1rem 0; font-size: 0.9rem;'>
                        Best for editing & version control
                    </p>
                </div>
                """, unsafe_allow_html=True)
                st.download_button(
                    "📥 DOWNLOAD MARKDOWN",
                    data=result.markdown_content,
                    file_name=Path(result.markdown_path).name if result.markdown_path else "newsletter.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="dl_md_main",
                    type="primary",
                )
        
        # File paths info
        st.markdown("---")
        with st.expander("📁 Saved Files & Details", expanded=False):
            if result.html_path:
                st.code(f"HTML: {result.html_path}")
            if result.markdown_path:
                st.code(f"Markdown: {result.markdown_path}")
            if result.run_json_path:
                st.code(f"Run data: {result.run_json_path}")
            if result.articles_xlsx_path:
                st.code(f"Articles: {result.articles_xlsx_path}")
            if result.summaries_json_path:
                st.code(f"Summaries: {result.summaries_json_path}")
            if getattr(result, "run_id", None):
                st.caption(f"Run ID: {result.run_id}")
        
        # Also show preview tabs
        st.markdown("### 👁️ Preview")
        tab_preview, tab_html, tab_md, tab_details = st.tabs(["Preview", "HTML", "Markdown", "Run details"])
        with tab_preview:
            if result.markdown_content:
                st.markdown(result.markdown_content)
            else:
                st.warning("No newsletter content was produced.")
        with tab_html:
            if result.html_content:
                st.code(result.html_content, language="html")
        with tab_md:
            if result.markdown_content:
                st.code(result.markdown_content, language="markdown")
        with tab_details:
            st.caption("Execution log")
            st.code("\n".join(result.execution_log or []))
            if result.error:
                st.error(result.error)
            critique = result.critique
            if critique is not None:
                st.write(f"Critic verdict: {'approved ✅' if critique.approved else 'changes requested ❌'}")
                if critique.improvement_instructions:
                    st.write("Improvement instructions:")
                    for instruction in critique.improvement_instructions:
                        st.write(f"- {instruction}")
            research_stats = final.get("research_stats")
            if research_stats is not None:
                st.caption("Research stats")
                st.json(research_stats.model_dump(mode="json"))
        
    elif result is not None:
        st.warning(f"Newsletter finished with issues — {len(result.summaries)} section(s). "
                   f"Tokens: {progress.input_tokens:,} in / {progress.output_tokens:,} out — {progress.elapsed:.1f}s total.")
    else:
        st.error("The run did not produce a result.")
        st.code("\n".join(final.get("errors") or []) or "No errors recorded; the graph ended early.")
        st.stop()

    # Reset success_shown flag when there's a new run
    if st.session_state.get("success_shown") and st.session_state.generating:
        st.session_state.success_shown = False