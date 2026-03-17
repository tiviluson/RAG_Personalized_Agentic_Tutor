"""Collection statistics dashboard."""

import httpx
import streamlit as st

from components.auth import require_auth

st.set_page_config(page_title="RAG Tutor - Dashboard", page_icon="", layout="wide")

API_BASE = "http://localhost:8000/api"

# ---------------------------------------------------------------------------
# Auth guard
# ---------------------------------------------------------------------------
require_auth()

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
    st.metric("Points", cc.get("points_count", 0))
    st.metric("Indexed Vectors", cc.get("indexed_vectors_count", 0))
    st.text(f"Status: {cc.get('status', 'unknown')}")

with col2:
    st.subheader("Student Notes")
    sn = stats.get("student_notes", {})
    st.metric("Points", sn.get("points_count", 0))
    st.metric("Indexed Vectors", sn.get("indexed_vectors_count", 0))
    st.text(f"Status: {sn.get('status', 'unknown')}")

if st.button("Refresh"):
    st.rerun()
