"""Keep the Render-hosted chatbot awake by pinging its health endpoint."""

import logging
import os
import sys

import httpx


logger = logging.getLogger(__name__)


def main() -> None:
    health_url = "https://backend-ai-yyof.onrender.com/chatbot/health"
    timeout = 300.0
    token = os.getenv("AI_API_TOKEN")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    headers = {"Authorization": f"Bearer {token}"} if token else None

    try:
        logger.info("Pinging %s", health_url)
        response = httpx.get(health_url, headers=headers, timeout=timeout)
        response.raise_for_status()
        logger.info("Healthcheck responded with %s", response.status_code)
    except Exception:
        logger.exception("Failed to ping Render health endpoint")
        sys.exit(1)


if __name__ == "__main__":
    main()
