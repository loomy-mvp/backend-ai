# Dockerfile for Cloud Run Job - Batch Embedding
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY kb/batch_embedding/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY kb/batch_embedding/src/ ./src/
COPY kb/batch_embedding/run_batch_embed.py .
COPY backend/ ./backend/

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "run_batch_embed.py"]

CMD ["--bucket-name", "bucket-org123", \
     "--folders", "organization/", \
     "--organization-id", "org123", \
     "--library", "organization", \
     "--log-level", "INFO"]
