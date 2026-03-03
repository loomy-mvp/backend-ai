"""Internal worker endpoints called by Google Cloud Tasks.

These endpoints are **not** protected by the standard API token
(``verify_token``). Instead they verify the OIDC token that Cloud Tasks
attaches to every dispatched request.

The router is mounted at ``/internal`` in ``main.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os

from fastapi import APIRouter, Depends, HTTPException, Request
from google.auth.transport import requests as google_auth_requests
from google.oauth2 import id_token

from backend.services.chatbot_api import process_chat_request
from backend.services.write_api import process_write_request

logger = logging.getLogger(__name__)

internal_router = APIRouter()


# ---------------------------------------------------------------------------
# Auth dependency – verify the OIDC token issued by Cloud Tasks
# ---------------------------------------------------------------------------

async def verify_cloud_tasks_token(request: Request):
    """Validate the OIDC bearer token set by Cloud Tasks.

    When ``CLOUD_TASKS_SA_EMAIL`` is **not** configured the check is skipped
    so that local development / testing still works.
    """

    sa_email = os.getenv("CLOUD_TASKS_SA_EMAIL")
    if not sa_email:
        # Auth disabled – allow (useful for local dev / curl testing)
        logger.debug("[verify_cloud_tasks_token] SA email not set; skipping verification")
        return

    authorization = request.headers.get("authorization", "")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")

    token = authorization.split(" ", 1)[1]

    try:
        loop = asyncio.get_event_loop()
        claims = await loop.run_in_executor(
            None,
            lambda: id_token.verify_oauth2_token(
                token,
                google_auth_requests.Request(),
            ),
        )
        logger.info(
            "[verify_cloud_tasks_token] Verified token – email=%s",
            claims.get("email"),
        )
        return claims
    except Exception as exc:
        logger.error("[verify_cloud_tasks_token] Token verification failed: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid Cloud Tasks OIDC token")


# ---------------------------------------------------------------------------
# Worker endpoint
# ---------------------------------------------------------------------------

@internal_router.post(
    "/process-chat",
    dependencies=[Depends(verify_cloud_tasks_token)],
)
async def worker_process_chat(request: Request):
    """Called by Cloud Tasks to execute the chat pipeline.

    Cloud Tasks expects a **2xx** response; any other status code triggers
    a retry according to the queue configuration.
    """
    chat_data: dict = await request.json()
    logger.info(
        "[worker_process_chat] Received task for message_id=%s",
        chat_data.get("message_id"),
    )

    # process_chat_request already catches exceptions internally and sends
    # error webhooks, so we do not need extra error handling here.
    await process_chat_request(chat_data)

    return {"status": "ok"}


@internal_router.post(
    "/process-write",
    dependencies=[Depends(verify_cloud_tasks_token)],
)
async def worker_process_write(request: Request):
    """Called by Cloud Tasks to execute the document writing pipeline."""
    write_data: dict = await request.json()
    logger.info(
        "[worker_process_write] Received task for message_id=%s",
        write_data.get("message_id"),
    )

    await process_write_request(write_data)

    return {"status": "ok"}
