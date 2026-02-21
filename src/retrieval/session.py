"""In-memory thread-safe session store for chat conversations."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from loguru import logger

from src.config import settings


def _now_local() -> str:
    """Return the current local time as a string."""
    return datetime.now().astimezone().isoformat()


@dataclass
class Turn:
    """A single conversation turn.

    Attributes:
        role: "user" or "assistant".
        content: The message text.
        citations: Optional citation metadata (for assistant turns).
        timestamp: When this turn was recorded.
    """

    role: str
    content: str
    citations: list[dict] | None = None
    timestamp: str = field(default_factory=_now_local)


@dataclass
class Session:
    """A chat session with conversation history.

    Attributes:
        session_id: Unique session identifier.
        student_id: The student who owns this session.
        course_id: The course context for this session.
        history: Ordered list of conversation turns.
        created_at: When the session was created.
        last_active: When the session was last used.
    """

    session_id: str
    student_id: str
    course_id: str
    history: list[Turn] = field(default_factory=list)
    created_at: str = field(default_factory=_now_local)
    last_active: str = field(default_factory=_now_local)


_store: dict[str, Session] = {}
_lock = threading.Lock()


def create_session(student_id: str, course_id: str) -> str:
    """Create a new chat session.

    Args:
        student_id: The student's identifier.
        course_id: The course identifier.

    Returns:
        The new session ID.
    """
    session_id = uuid.uuid4().hex
    session = Session(
        session_id=session_id,
        student_id=student_id,
        course_id=course_id,
    )
    with _lock:
        _store[session_id] = session
    logger.info("Created session {} for student {}", session_id, student_id)
    return session_id


def get_session(session_id: str) -> Session | None:
    """Retrieve a session by ID.

    Args:
        session_id: The session identifier.

    Returns:
        The Session object, or None if not found.
    """
    with _lock:
        return _store.get(session_id)


def add_turn(
    session_id: str,
    role: str,
    content: str,
    citations: list[dict] | None = None,
) -> None:
    """Append a conversation turn to a session.

    Args:
        session_id: The session identifier.
        role: "user" or "assistant".
        content: The message text.
        citations: Optional citation metadata.
    """
    with _lock:
        session = _store.get(session_id)
        if session is None:
            return
        session.history.append(
            Turn(role=role, content=content, citations=citations)
        )
        session.last_active = _now_local()


def get_recent_history(
    session_id: str,
    max_turns: int | None = None,
) -> list[dict]:
    """Get the most recent turns as plain dicts.

    Args:
        session_id: The session identifier.
        max_turns: Maximum number of turns to return.
            Defaults to ``settings.history_max_turns``.

    Returns:
        List of dicts with ``role`` and ``content`` keys.
    """
    if max_turns is None:
        max_turns = settings.history_max_turns

    with _lock:
        session = _store.get(session_id)
        if session is None:
            return []
        recent = session.history[-max_turns:]
        return [{"role": t.role, "content": t.content} for t in recent]


def close_session(session_id: str) -> bool:
    """Remove a session immediately.

    Args:
        session_id: The session identifier.

    Returns:
        True if the session existed and was removed.
    """
    with _lock:
        if session_id in _store:
            del _store[session_id]
            logger.info("Closed session {}", session_id)
            return True
        return False


def cleanup_expired_sessions(max_age_hours: int | None = None) -> int:
    """Remove sessions that have been inactive beyond the TTL.

    Args:
        max_age_hours: Maximum inactivity in hours.
            Defaults to ``settings.session_ttl_hours``.

    Returns:
        Number of sessions removed.
    """
    if max_age_hours is None:
        max_age_hours = settings.session_ttl_hours

    cutoff = datetime.now().astimezone() - timedelta(hours=max_age_hours)
    with _lock:
        to_remove = [
            sid
            for sid, session in _store.items()
            if datetime.fromisoformat(session.last_active) < cutoff
        ]
        for sid in to_remove:
            del _store[sid]
    if to_remove:
        logger.info("Cleaned up {} expired sessions", len(to_remove))
    return len(to_remove)
