# Dockerfile for Cloud Run Job
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY kb/scraping/agenzia-entrate-interpelli/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the scraping module with correct paths
COPY kb/scraping/agenzia-entrate-interpelli/src/ ./src/
COPY kb/scraping/utils/ ./src/utils/
COPY kb/scraping/agenzia-entrate-interpelli/run_scraper.py .

# Set Python path
ENV PYTHONPATH=/app

# Use ENTRYPOINT to allow passing arguments
ENTRYPOINT ["python", "run_scraper.py"]

# Default arguments (can be overridden at runtime)
CMD ["--bucket-name", "loomy-public-documents", \
     "--base-folder", "interpelli", \
     "--max-concurrent", "10", \
     "--log-level", "INFO"]
