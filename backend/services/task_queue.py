"""Google Cloud Tasks integration for async chat processing.

Instead of running chat requests in-process via BackgroundTasks, this module
enqueues them as Cloud Tasks that call back into the ``/internal/process-chat``
worker endpoint.  Cloud Tasks handles concurrency throttling, automatic retries
and persistence.

Required environment variables
-------------------------------
GCP_PROJECT_ID              – Google Cloud project ID
GCP_LOCATION                – Queue region (default ``europe-west1``)
CLOUD_TASKS_QUEUE           – Queue name  (default ``chat-queue``)
WORKER_BASE_URL             – Public base URL of this server (e.g. ``https://my-app.onrender.com``)
CLOUD_TASKS_SA_EMAIL        – (Optional) Service-account email for OIDC token
"""

from __future__ import annotations

import json
import logging
import os

from google.cloud import tasks_v2
from google.protobuf import timestamp_pb2

logger = logging.getLogger(__name__)


def _get_env(key: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(key, default)
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


def enqueue_chat_task(chat_data: dict) -> tasks_v2.Task:
    """Create a Cloud Task that POSTs *chat_data* to the worker endpoint.

    Returns the created ``Task`` proto so callers can inspect ``task.name``
    if needed.
    """

    project = _get_env("GCP_PROJECT_ID", required=True)
    location = _get_env("GCP_LOCATION", default="europe-west1")
    queue = _get_env("CLOUD_TASKS_QUEUE", default="chat-queue")
    worker_base_url = _get_env("WORKER_BASE_URL", required=True)
    sa_email = _get_env("CLOUD_TASKS_SA_EMAIL")

    client = tasks_v2.CloudTasksClient()
    parent = client.queue_path(project, location, queue)

    target_url = f"{worker_base_url.rstrip('/')}/internal/process-chat"

    http_request: dict = {
        "http_method": tasks_v2.HttpMethod.POST,
        "url": target_url,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(chat_data).encode(),
    }

    # Attach an OIDC token so the worker can verify the caller
    if sa_email:
        http_request["oidc_token"] = {
            "service_account_email": sa_email,
            "audience": target_url,
        }

    task: dict = {"http_request": http_request}

    response = client.create_task(request={"parent": parent, "task": task})
    logger.info("[enqueue_chat_task] Task created: %s", response.name)
    return response
