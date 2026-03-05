"""Chat interface with answer modes, citations, and filter sidebar."""

import json
import sys
from pathlib import Path

import httpx
import streamlit as st
from loguru import logger

# Add project root to path so Streamlit pages can import src.*
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

st.set_page_config(page_title="RAG Tutor - Chat", layout="wide")

API_BASE = "http://localhost:8000/api"

# ---------------------------------------------------------------------------
# Auth guard
# ---------------------------------------------------------------------------
if not st.session_state.get("user_email"):
    st.warning("Please sign in on the home page first.")
    st.stop()

user_email = st.session_state["user_email"]
role = st.session_state.get("role", "student")

# ---------------------------------------------------------------------------
# Sidebar: answer mode + filters
# ---------------------------------------------------------------------------
st.sidebar.header("Chat Settings")

answer_mode = st.sidebar.radio(
    "Answer Mode",
    options=["long", "short", "eli5"],
    format_func=lambda m: {"long": "Long (detailed)", "short": "Short (concise)", "eli5": "ELI5 (simple)"}[m],
    index=0,
)

with st.sidebar.expander("Filters", expanded=False):
    filter_course_id = st.text_input("Course ID", value="", key="filter_course_id")
    filter_module_week = st.text_input("Module Week", value="", key="filter_module_week")
    filter_module_name = st.text_input("Module Name", value="", key="filter_module_name")
    filter_uploaded_by = st.text_input("Uploaded By", value="", key="filter_uploaded_by")
    filter_source_filename = st.text_input("Source Filename", value="", key="filter_source_filename")


def _build_filters() -> dict | None:
    """Build a filters dict from sidebar inputs, or None if all empty."""
    filters = {}
    if filter_course_id:
        filters["course_id"] = filter_course_id
    if filter_module_week:
        filters["module_week"] = filter_module_week
    if filter_module_name:
        filters["module_name"] = filter_module_name
    if filter_uploaded_by:
        filters["uploaded_by"] = filter_uploaded_by
    if filter_source_filename:
        filters["source_filename"] = filter_source_filename
    return filters or None


# ---------------------------------------------------------------------------
# Session initialization
# ---------------------------------------------------------------------------
if "chat_session_id" not in st.session_state:
    st.session_state["chat_session_id"] = None
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []
if "chat_citations" not in st.session_state:
    st.session_state["chat_citations"] = {}


def _create_session() -> str | None:
    """Create a chat session via the API and return the session_id."""
    try:
        resp = httpx.post(
            f"{API_BASE}/chat/session",
            json={"student_id": user_email, "course_id": filter_course_id or "default"},
            timeout=10,
        )
        resp.raise_for_status()
        sid = resp.json()["session_id"]
        logger.info("Session created: {} for user {}", sid, user_email)
        return sid
    except Exception as e:
        st.error(f"Failed to create session: {e}")
        logger.error("Session creation failed: {}", e)
        return None


# Auto-create session on first visit
if st.session_state["chat_session_id"] is None:
    sid = _create_session()
    if sid:
        st.session_state["chat_session_id"] = sid

if st.session_state["chat_session_id"] is None:
    st.error("Could not establish a chat session. Is the API running?")
    st.stop()

# ---------------------------------------------------------------------------
# Display conversation history
# ---------------------------------------------------------------------------
st.title("Chat")

for i, msg in enumerate(st.session_state["chat_messages"]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show citations for assistant messages
        msg_citations = st.session_state["chat_citations"].get(i)
        if msg_citations:
            with st.expander(f"Sources ({len(msg_citations)} references)", expanded=False):
                for cite in msg_citations:
                    label = f"[{cite['index']}] {cite.get('source_filename', 'Unknown')}"
                    page = cite.get("page_num")
                    if page:
                        label += f", p.{page}"
                    section = cite.get("section")
                    if section:
                        label += f" - {section}"
                    score = cite.get("relevance_score", 0)
                    st.markdown(
                        f"**{label}** (score: {score:.3f})  \n"
                        f"_{cite.get('text_preview', '')}_"
                    )

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ask a question about the course..."):
    # Display user message
    st.session_state["chat_messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        citations = []

        try:
            body = {
                "session_id": st.session_state["chat_session_id"],
                "query": prompt,
                "answer_mode": answer_mode,
            }
            filters = _build_filters()
            if filters:
                body["filters"] = filters

            logger.info(
                "Query submitted: session={}, mode={}, filters={}, query={!r}",
                st.session_state["chat_session_id"],
                answer_mode,
                filters,
                prompt[:120],
            )

            with httpx.stream(
                "POST",
                f"{API_BASE}/chat/query",
                json=body,
                timeout=httpx.Timeout(connect=10, read=300, write=10, pool=10),
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = json.loads(line[6:])
                    event_type = payload.get("type")

                    if event_type == "status":
                        response_placeholder.markdown(f"*{payload.get('data', '')}*")

                    elif event_type == "chunk":
                        full_response += payload.get("data", "")
                        response_placeholder.markdown(full_response + " |")

                    elif event_type == "done":
                        done_data = payload.get("data", "{}")
                        if isinstance(done_data, str):
                            done_data = json.loads(done_data)
                        citations = done_data.get("citations", [])
                        metrics = done_data.get("metrics", {})
                        logger.info(
                            "Response complete: {} citations, metrics={}",
                            len(citations),
                            metrics,
                        )

                    elif event_type == "error":
                        error_msg = payload.get("data", "Unknown error")
                        logger.error("Server-side pipeline error: {}", error_msg)
                        st.error(f"Error: {error_msg}")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning("Session {} expired, resetting", st.session_state["chat_session_id"])
                st.error("Session expired. Refreshing...")
                st.session_state["chat_session_id"] = None
                st.rerun()
            else:
                logger.error("API error {}: {}", e.response.status_code, e)
                st.error(f"API error: {e}")
                full_response = "Sorry, an error occurred."
        except Exception as e:
            logger.error("Connection error: {}", e)
            st.error(f"Connection error: {e}")
            full_response = "Sorry, could not reach the server."

        # Final render (remove cursor)
        response_placeholder.markdown(full_response)

    # Store assistant message
    msg_index = len(st.session_state["chat_messages"])
    st.session_state["chat_messages"].append({"role": "assistant", "content": full_response})
    if citations:
        st.session_state["chat_citations"][msg_index] = citations

        # Show citations inline
        with st.expander(f"Sources ({len(citations)} references)", expanded=False):
            for cite in citations:
                label = f"[{cite['index']}] {cite.get('source_filename', 'Unknown')}"
                page = cite.get("page_num")
                if page:
                    label += f", p.{page}"
                section = cite.get("section")
                if section:
                    label += f" - {section}"
                score = cite.get("relevance_score", 0)
                st.markdown(
                    f"**{label}** (score: {score:.3f})  \n"
                    f"_{cite.get('text_preview', '')}_"
                )
