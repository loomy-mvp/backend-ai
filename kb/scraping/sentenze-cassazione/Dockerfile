FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY kb/scraping/sentenze-cassazione/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY kb/scraping/sentenze-cassazione/src/ ./src/

# Set environment variable for Cloud Run
ENV PYTHONUNBUFFERED=1

# Run the job
ENTRYPOINT ["python", "src/cloudrun_job_scraper.py"]
