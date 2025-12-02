# Container image for the Bedrock batch processing job
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY backend/jobs/bedrock_batch_process/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/jobs/bedrock_batch_process/bedrock_batch_process.py ./bedrock_batch_process.py

# Use ENTRYPOINT to allow passing arguments
ENTRYPOINT ["python", "bedrock_batch_process.py"]

# Default arguments (can be overridden at runtime)
CMD ["--bucket", "loomy-jobs", \
     "--model", "eu.amazon.nova-micro-v1:0"]
