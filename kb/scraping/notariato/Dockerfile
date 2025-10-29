# Dockerfile for Cloud Run Job
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY notariato/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the scraping module with correct paths
COPY notariato/src/ ./src/
COPY utils/ ./src/utils/
COPY notariato/run_scraper.py .

# Set Python path
ENV PYTHONPATH=/app

# Use ENTRYPOINT to allow passing arguments
ENTRYPOINT ["python", "run_scraper.py"]

# Default arguments (can be overridden at runtime)
CMD ["--bucket-name", "loomy-public-documents", \
     "--base-folder", "notariato", \
     "--max-concurrent", "10", \
     "--log-level", "INFO"]
