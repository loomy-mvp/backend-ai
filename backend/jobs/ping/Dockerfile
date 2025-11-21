# Container image for the Render keep-alive job
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY backend/jobs/ping/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/jobs/ping/keep_render_alive.py ./keep_render_alive.py

CMD ["python", "keep_render_alive.py"]
