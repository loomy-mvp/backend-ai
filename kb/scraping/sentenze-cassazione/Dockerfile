# Dockerfile for Cloud Run Job
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY kb/scraping/sentenze-cassazione/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy scraper sources (module + shared utils)
COPY kb/scraping/sentenze-cassazione/src/ ./src/
COPY kb/scraping/utils/ ./src/utils/
COPY kb/scraping/sentenze-cassazione/run_scraper.py ./

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "run_scraper.py"]
CMD ["--bucket-name", "loomy-public-documents", \
     "--base-folder", "sentenze_cassazione", \
     "--request-delay", "0.5", \
     "--log-level", "INFO"]
