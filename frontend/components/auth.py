"""Shared authentication guard for all Streamlit pages."""

import streamlit as st
from loguru import logger


def require_auth() -> tuple[str, str, str]:
    """Check authentication and return user info, or stop the page.

    Uses Streamlit's built-in ``st.user`` (cookie-based) so that a browser
    refresh on any page re-authenticates automatically without redirecting
    to the Homepage.

    Returns:
        tuple[str, str, str]: (user_email, user_name, role)
    """
    if not st.user.is_logged_in:
        st.title("RAG Collaborative Personalized Tutor")
        st.write("Please sign in to continue.")
        st.button("Sign in with Google", on_click=st.login)
        st.stop()

    user_email = st.user.email or ""
    user_name = st.user.name or ""

    # Populate session state so downstream code can still read it
    if "user_email" not in st.session_state or st.session_state["user_email"] != user_email:
        logger.info("User signed in: {} ({})", user_name, user_email)
        lecturer_emails = st.secrets.get("lecturer_emails", [])
        role = "lecturer" if user_email in lecturer_emails else "student"
        st.session_state["user_email"] = user_email
        st.session_state["user_name"] = user_name
        st.session_state["role"] = role

    return (
        st.session_state["user_email"],
        st.session_state["user_name"],
        st.session_state["role"],
    )
