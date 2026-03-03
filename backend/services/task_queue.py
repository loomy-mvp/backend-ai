"""Google Cloud Tasks integration for async processing.

Instead of running requests in-process via BackgroundTasks, this module
enqueues them as Cloud Tasks that call back into ``/internal/*`` worker
endpoints.  Cloud Tasks handles concurrency throttling, automatic retries
and persistence.

Required environment variables
-------------------------------
GCP_PROJECT_ID              – Google Cloud project ID
GCP_LOCATION                – Queue region (default ``europe-west1``)
CLOUD_TASKS_QUEUE           – Queue name  (default ``chat-queue``)
WORKER_BASE_URL             – Public base URL of this server (e.g. ``https://my-app.onrender.com``)
GOOGLE_CLOUD_TASKS_CREDENTIALS_JSON – Full content of the Cloud Tasks service account JSON key file
CLOUD_TASKS_SA_EMAIL        – (Optional) Service-account email for OIDC token
"""

from __future__ import annotations

import json
import logging
import os

from google.cloud import tasks_v2
from google.oauth2 import service_account

logger = logging.getLogger(__name__)


def _get_env(key: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(key, default)
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


def _get_credentials():
    """Load GCP credentials from GOOGLE_CREDENTIALS_JSON env var (Render-friendly).

    Falls back to Application Default Credentials when the env var is absent
    (works on local dev with ``gcloud auth application-default login``).
    """
    creds_json = os.getenv("GOOGLE_CLOUD_TASKS_CREDENTIALS_JSON")
    if not creds_json:
        return None  # let the client library use ADC

    info = json.loads(creds_json)
    return service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/cloud-tasks"],
    )


def _enqueue_task(endpoint_path: str, payload: dict, label: str = "task") -> tasks_v2.Task:
    """Generic helper – enqueue a Cloud Task targeting *endpoint_path*."""

    project = _get_env("GCP_PROJECT_ID", required=True)
    location = _get_env("GCP_LOCATION", default="europe-west1")
    queue = _get_env("CLOUD_TASKS_QUEUE", default="chat-queue")
    worker_base_url = _get_env("WORKER_BASE_URL", required=True)
    sa_email = _get_env("CLOUD_TASKS_SA_EMAIL")

    credentials = _get_credentials()
    client = tasks_v2.CloudTasksClient(credentials=credentials)
    parent = client.queue_path(project, location, queue)

    target_url = f"{worker_base_url.rstrip('/')}{endpoint_path}"

    http_request: dict = {
        "http_method": tasks_v2.HttpMethod.POST,
        "url": target_url,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload).encode(),
    }

    # Attach an OIDC token so the worker can verify the caller
    if sa_email:
        http_request["oidc_token"] = {
            "service_account_email": sa_email,
            "audience": target_url,
        }

    task: dict = {"http_request": http_request}

    response = client.create_task(request={"parent": parent, "task": task})
    logger.info("[%s] Task created: %s", label, response.name)
    return response


def enqueue_chat_task(chat_data: dict) -> tasks_v2.Task:
    """Enqueue a chat processing task."""
    return _enqueue_task("/internal/process-chat", chat_data, label="enqueue_chat_task")


def enqueue_write_task(write_data: dict) -> tasks_v2.Task:
    """Enqueue a document writing task."""
    return _enqueue_task("/internal/process-write", write_data, label="enqueue_write_task")
