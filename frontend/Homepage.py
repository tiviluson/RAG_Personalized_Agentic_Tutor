"""Streamlit entry point with Google Sign-In authentication."""

import streamlit as st

from components.auth import require_auth

st.set_page_config(
    page_title="RAG Tutor",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Google Sign-In (shared auth guard)
# ---------------------------------------------------------------------------
user_email, user_name, role = require_auth()

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
st.title("RAG Collaborative Personalized Tutor")
st.write("Use the sidebar to navigate to **Upload**, **Chat**, or **Dashboard**.")
