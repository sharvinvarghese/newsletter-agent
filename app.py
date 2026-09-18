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

settings = get_settings()

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
    provider = st.selectbox(
        "LLM Provider",
        options=["openrouter", "groq"],
        index=0 if settings.provider != "groq" else 1,
        format_func=lambda p: "OpenRouter" if p == "openrouter" else "Groq (fast)",
    )
    
    # Load saved model config for smart defaults
    model_config = load_model_config()
    saved_model = model_config.get("last_working_model", "")
    saved_provider = model_config.get("last_working_provider", "")
    
    # Use saved model as default if it matches current provider
    if provider == "groq":
        default_model = saved_model if saved_provider == "groq" and saved_model else settings.groq_model
        model = st.text_input("Model (Groq slug)", value=default_model)
    else:
        default_model = saved_model if saved_provider == "openrouter" and saved_model else settings.openrouter_model
        model = st.text_input("Model (OpenRouter slug)", value=default_model)
    
    target = st.number_input("Target articles", min_value=1, max_value=8, value=min(settings.target_articles, 2))
    window = st.number_input("Recency window (days)", min_value=0, max_value=60, value=min(settings.recency_window_days, 14))
    revisions = st.number_input("Max revisions", min_value=0, max_value=1, value=min(settings.max_revisions, 1))
    
    # Save intermediate results option
    save_intermediate = st.checkbox(
        "Save intermediate results",
        value=False,
        help="Save all artifacts (run.json, articles.xlsx, summaries.json) except newsletter.html"
    )

    st.divider()
    _has_key = settings.has_groq if provider == "groq" else settings.has_api_key
    run_clicked = st.button(
        "Generate newsletter",
        type="primary",
        use_container_width=True,
        disabled=(not _has_key or st.session_state.generating),
    )
    if not _has_key:
        st.warning(f"{'GROQ_API_KEY' if provider == 'groq' else 'OPENROUTER_API_KEY'} not configured; button disabled.")


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
        st.markdown("### Newsletter progress")
        _render_node_stepper(progress)
        st.markdown("---")
        total = len(NODE_ORDER)
        elapsed = progress.elapsed
        idx = len(progress.completed_nodes)
        frac = idx / total if total else 0.0
        st.progress(min(frac, 1.0), text=progress.status or f"{progress.last_node or 'starting'} …")
        st.caption(f"Step {idx} of {total} — {progress.last_node or '—'} — {elapsed:.1f}s elapsed")
        if progress.input_tokens or progress.output_tokens:
            st.caption(f"Tokens — {progress.input_tokens:,} in / {progress.output_tokens:,} out")
        if progress.raw_article_count or progress.article_count:
            st.write("**Fetched articles**")
            st.write(f"- Collected: {progress.raw_article_count}")
            st.write(f"- After dedupe/selection: {progress.article_count}")


def _run(*, on_progress=None):
    from agent.graph import run_newsletter_state

    # Speed profile: fewer articles, tighter timeouts, fewer sources per run.
    overrides = {
        "target_articles": min(int(target), 6),
        "max_articles_to_collect": 18,
        "recency_window_days": int(window),
        "max_revisions": min(int(revisions), 1),
        "research_max_sources": 4,
        "max_items_per_source": 8,
        "enrich_article_limit": 6,
        "request_timeout_seconds": 10.0,
        "research_max_workers": 4,
        "llm_timeout_seconds": 30.0,
        "provider": provider,
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
    _has_key = settings.has_groq if provider == "groq" else settings.has_api_key
    if not _has_key:
        st.error(f"{'GROQ_API_KEY' if provider == 'groq' else 'OPENROUTER_API_KEY'} not configured. Add it to newsletter-agent/.env and restart Streamlit.")
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
    st.caption("Generating newsletter…")
    st.rerun()

final = st.session_state.get("last_run")
if isinstance(final, dict) and final:
    progress = st.session_state.run_progress
    result = final.get("final_output")
    if result is not None and result.success:
        # Save the working model for next run (smart switch)
        if model.strip():
            save_model_config({
                "last_working_model": model.strip(),
                "last_working_provider": provider,
            })
        
        # Success toast with file paths
        st.toast(f"✅ Newsletter saved! HTML: {Path(result.html_path).name if result.html_path else 'N/A'}", icon="🎉")
        
        st.success(f"Newsletter complete — {len(result.summaries)} section(s), {result.revision_count} revision(s). "
                 f"Tokens: {progress.input_tokens:,} in / {progress.output_tokens:,} out — {progress.elapsed:.1f}s total.")
    elif result is not None:
        st.warning(f"Newsletter finished with issues — {len(result.summaries)} section(s). "
                   f"Tokens: {progress.input_tokens:,} in / {progress.output_tokens:,} out — {progress.elapsed:.1f}s total.")
    else:
        st.error("The run did not produce a result.")
        st.code("\n".join(final.get("errors") or []) or "No errors recorded; the graph ended early.")
        st.stop()

    newsletter = result.newsletter
    if newsletter is not None:
        st.subheader(newsletter.subject)
        st.caption(newsletter.preheader)

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

    left, right = st.columns(2)
    if result.html_content:
        left.download_button(
            "Download HTML",
            data=result.html_content,
            file_name=Path(result.html_path).name if result.html_path else "newsletter.html",
            mime="text/html",
            use_container_width=True,
        )
    if result.markdown_content:
        right.download_button(
            "Download Markdown",
            data=result.markdown_content,
            file_name=Path(result.markdown_path).name if result.markdown_path else "newsletter.md",
            mime="text/markdown",
            use_container_width=True,
        )
    if result.html_path:
        st.caption(f"Saved: {result.html_path}")
    if result.markdown_path:
        st.caption(f"Saved: {result.markdown_path}")
    if result.run_json_path:
        st.caption(f"Run artifact: {result.run_json_path}")
    if result.articles_xlsx_path:
        st.caption(f"Articles: {result.articles_xlsx_path}")
    if result.summaries_json_path:
        st.caption(f"Summaries: {result.summaries_json_path}")
    if getattr(result, "run_id", None):
        st.caption(f"Run ID: {result.run_id}")
