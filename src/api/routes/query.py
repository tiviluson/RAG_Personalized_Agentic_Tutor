"""Chat and query endpoints with SSE streaming."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse

from src.api.models.query import (
    ChatQueryRequest,
    CreateSessionRequest,
    CreateSessionResponse,
)
from src.retrieval.pipeline import run_pipeline
from src.retrieval.session import close_session, create_session, get_session

router = APIRouter()


@router.post("/chat/session", response_model=CreateSessionResponse)
def create_chat_session(body: CreateSessionRequest) -> CreateSessionResponse:
    """Create a new chat session.

    Args:
        body: Request with student_id and course_id.

    Returns:
        The new session ID.
    """
    session_id = create_session(body.student_id, body.course_id)
    return CreateSessionResponse(session_id=session_id)


@router.post("/chat/query")
async def chat_query(body: ChatQueryRequest) -> StreamingResponse:
    """Stream a chat response via SSE.

    Args:
        body: Request with session_id, query, answer_mode, and optional filters.

    Returns:
        SSE stream of ChatEvent objects.

    Raises:
        HTTPException: 404 if the session does not exist.
    """
    session = get_session(body.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    filters = body.filters.model_dump(exclude_none=True) if body.filters else None

    async def event_stream():
        async for event in run_pipeline(
            session_id=body.session_id,
            query=body.query,
            answer_mode=body.answer_mode,
            filters=filters,
        ):
            yield f"data: {json.dumps(event.model_dump())}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete("/chat/session/{session_id}", status_code=204)
def delete_chat_session(session_id: str) -> Response:
    """Close and remove a chat session.

    Args:
        session_id: The session to close.

    Returns:
        204 No Content on success.

    Raises:
        HTTPException: 404 if the session does not exist.
    """
    removed = close_session(session_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Session not found")
    return Response(status_code=204)
