FROM python:3.11-slim

WORKDIR /app

COPY kb/scraping/agevolazioni/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY kb/scraping/agevolazioni/src/ ./src/
COPY kb/scraping/utils/ ./src/utils/
COPY kb/scraping/agevolazioni/run_scraper.py ./

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "run_scraper.py"]
CMD ["--bucket-name", "loomy-public-documents", \
     "--base-folder", "agevolazioni", \
     "--max-concurrent", "5", \
     "--log-level", "INFO"]
