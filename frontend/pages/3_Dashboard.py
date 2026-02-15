"""Collection statistics dashboard."""

import httpx
import streamlit as st

st.set_page_config(page_title="RAG Tutor - Dashboard", page_icon="", layout="wide")

API_BASE = "http://localhost:8000/api"

# ---------------------------------------------------------------------------
# Auth guard
# ---------------------------------------------------------------------------
if not st.session_state.get("user_email"):
    st.warning("Please sign in on the home page first.")
    st.stop()

st.title("Collection Dashboard")

# ---------------------------------------------------------------------------
# Fetch stats
# ---------------------------------------------------------------------------
try:
    with httpx.Client(timeout=10) as client:
        resp = client.get(f"{API_BASE}/collections/stats")
        resp.raise_for_status()
        stats = resp.json()
except httpx.ConnectError:
    st.error("Cannot connect to API server. Is it running?")
    st.stop()
except httpx.HTTPStatusError as e:
    st.error(f"Failed to fetch stats: {e.response.text}")
    st.stop()

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Course Content")
    cc = stats.get("course_content", {})
    st.metric("Points", cc.get("points_count", "N/A"))
    st.text(f"Status: {cc.get('status', 'unknown')}")

with col2:
    st.subheader("Student Notes")
    sn = stats.get("student_notes", {})
    st.metric("Points", sn.get("points_count", "N/A"))
    st.text(f"Status: {sn.get('status', 'unknown')}")

if st.button("Refresh"):
    st.rerun()
