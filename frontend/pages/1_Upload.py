"""Document upload page with progress tracking."""

import time

import sys
from pathlib import Path

import httpx
import streamlit as st
from loguru import logger

# Add project root to path so Streamlit pages can import src.*
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from components.auth import require_auth
from src.ingestion.types import DocType

st.set_page_config(page_title="RAG Tutor - Upload", page_icon="", layout="wide")

API_BASE = "http://localhost:8000/api"
POLL_INTERVAL_S = 2

# ---------------------------------------------------------------------------
# Auth guard
# ---------------------------------------------------------------------------
user_email, _, role = require_auth()

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
        "Document type (applies to all files)",
        options=[dt.value for dt in DocType],
        format_func=lambda v: v.replace("_", " ").title(),
    )

    course_id = st.text_input("Course ID", value="")

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

    form_data: dict[str, str | list[str]] = {
        "role": role,
        "course_id": course_id,
        "module_name": module_name,
        "module_week": str(module_week) if module_week else "",
        "doc_type": doc_type,
    }
    if student_id:
        form_data["student_id"] = student_id

    filenames = [f.name for f in files]
    logger.info("Uploading {} file(s): {}", len(files), filenames)

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
        logger.error("Upload failed: {}", e.response.text)
        st.error(f"Upload failed: {e.response.text}")
        st.stop()
    except httpx.ConnectError:
        logger.error("Cannot connect to API server")
        st.error("Cannot connect to API server. Is it running?")
        st.stop()

    jobs = result.get("jobs", [])
    if not jobs:
        st.warning("No jobs returned.")
        st.stop()

    logger.info("Queued {} job(s): {}", len(jobs), [j["job_id"] for j in jobs])
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
                        logger.error("Job {} ({}) failed: {}", job_id, ctx["filename"], error)
                        ctx["status_text"].text(f"Failed: {error}")
                    else:
                        chunks = data.get("chunks_indexed", 0)
                        logger.info("Job {} ({}) complete: {} chunks indexed", job_id, ctx["filename"], chunks)
                        ctx["status_text"].text(
                            f"Complete -- {chunks} chunks indexed"
                        )

            time.sleep(POLL_INTERVAL_S)

elif submitted and not files:
    st.warning("Please select at least one file.")
