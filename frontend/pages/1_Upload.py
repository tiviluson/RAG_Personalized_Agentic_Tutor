"""Document upload page with progress tracking."""

import time

import sys
from pathlib import Path

import httpx
import streamlit as st

# Add project root to path so Streamlit pages can import src.*
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion.types import DocType

st.set_page_config(page_title="RAG Tutor - Upload", page_icon="", layout="wide")

API_BASE = "http://localhost:8000/api"
POLL_INTERVAL_S = 2

# ---------------------------------------------------------------------------
# Auth guard
# ---------------------------------------------------------------------------
if not st.session_state.get("user_email"):
    st.warning("Please sign in on the home page first.")
    st.stop()

role = st.session_state.get("role", "student")
user_email = st.session_state["user_email"]

st.title("Upload Documents")

# ---------------------------------------------------------------------------
# Upload form
# ---------------------------------------------------------------------------
with st.form("upload_form"):
    files = st.file_uploader(
        "Select files (.pdf or .md)",
        type=["pdf", "md"],
        accept_multiple_files=True,
    )

    doc_type = st.selectbox(
        "Document type",
        options=[dt.value for dt in DocType],
        format_func=lambda v: v.replace("_", " ").title(),
    )

    course_id = st.number_input("Course ID", min_value=1, step=1, value=1)

    col1, col2 = st.columns(2)
    with col1:
        module_name = st.text_input("Module name (optional)")
    with col2:
        module_week = st.number_input(
            "Module week (optional)", min_value=0, step=1, value=0
        )

    # Student ID is auto-filled from session for students
    if role == "student":
        student_id = user_email
        st.text_input("Student ID", value=student_id, disabled=True)
    else:
        student_id = None

    submitted = st.form_submit_button("Upload")

# ---------------------------------------------------------------------------
# Handle submission
# ---------------------------------------------------------------------------
if submitted and files:
    upload_files = [
        ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
        for f in files
    ]

    form_data = {
        "doc_type": doc_type,
        "role": role,
        "course_id": str(course_id),
        "module_name": module_name,
        "module_week": str(module_week) if module_week else "",
    }
    if student_id:
        form_data["student_id"] = student_id

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{API_BASE}/upload",
                files=upload_files,
                data=form_data,
            )
            resp.raise_for_status()
            result = resp.json()
    except httpx.HTTPStatusError as e:
        st.error(f"Upload failed: {e.response.text}")
        st.stop()
    except httpx.ConnectError:
        st.error("Cannot connect to API server. Is it running?")
        st.stop()

    jobs = result.get("jobs", [])
    if not jobs:
        st.warning("No jobs returned.")
        st.stop()

    st.success(f"Queued {len(jobs)} file(s) for processing.")

    # -------------------------------------------------------------------
    # Progress polling
    # -------------------------------------------------------------------
    status_containers = {}
    for job in jobs:
        job_id = job["job_id"]
        filename = job["filename"]
        container = st.container()
        with container:
            st.subheader(filename)
            progress_bar = st.progress(0)
            status_text = st.empty()
        status_containers[job_id] = {
            "filename": filename,
            "progress_bar": progress_bar,
            "status_text": status_text,
            "done": False,
        }

    with httpx.Client(timeout=10) as client:
        while not all(c["done"] for c in status_containers.values()):
            for job_id, ctx in status_containers.items():
                if ctx["done"]:
                    continue
                try:
                    resp = client.get(f"{API_BASE}/upload/status/{job_id}")
                    resp.raise_for_status()
                    data = resp.json()
                except Exception:
                    continue

                status = data.get("status", "unknown")
                progress = data.get("progress", 0)
                ctx["progress_bar"].progress(progress)
                ctx["status_text"].text(f"Status: {status} ({progress}%)")

                if status in ("complete", "failed"):
                    ctx["done"] = True
                    if status == "failed":
                        error = data.get("error", "Unknown error")
                        ctx["status_text"].text(f"Failed: {error}")
                    else:
                        chunks = data.get("chunks_indexed", 0)
                        ctx["status_text"].text(
                            f"Complete -- {chunks} chunks indexed"
                        )

            time.sleep(POLL_INTERVAL_S)

elif submitted and not files:
    st.warning("Please select at least one file.")
