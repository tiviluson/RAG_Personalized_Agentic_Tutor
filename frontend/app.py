"""Streamlit entry point with Google Sign-In authentication."""

import streamlit as st
from loguru import logger

st.set_page_config(
    page_title="RAG Tutor",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Google Sign-In (Streamlit built-in auth)
# ---------------------------------------------------------------------------
if not st.user.is_logged_in:
    st.title("RAG Personalized Agentic Tutor")
    st.write("Please sign in to continue.")
    st.button("Sign in with Google", on_click=st.login)
    st.stop()

# User is authenticated
user_email = st.user.email or ""
user_name = st.user.name or ""
logger.info("User signed in: {} ({})", user_name, user_email)

# Role determination
lecturer_emails = st.secrets.get("lecturer_emails", [])
role = "lecturer" if user_email in lecturer_emails else "student"

st.session_state["user_email"] = user_email
st.session_state["user_name"] = user_name
st.session_state["role"] = role

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("RAG Tutor")
st.sidebar.write(f"Signed in as: **{user_name}**")
st.sidebar.write(f"Role: **{role}**")
st.sidebar.write(f"Email: {user_email}")
st.sidebar.button("Sign out", on_click=st.logout)

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("RAG Personalized Agentic Tutor")
st.write("Use the sidebar to navigate to **Upload**, **Chat**, or **Dashboard**.")
