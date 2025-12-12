FROM python:3.11-slim

# Install LibreOffice for DOC to PDF conversion
RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice-writer \
    libreoffice-common \
    fonts-liberation \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY kb/scraping/principi-contabili/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY kb/scraping/principi-contabili/src/ ./src/
COPY kb/scraping/utils/ ./src/utils/
COPY kb/scraping/principi-contabili/run_scraper.py ./

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "run_scraper.py"]
CMD ["--bucket-name", "loomy-public-documents", \
     "--base-folder", "principi-contabili", \
     "--max-concurrent", "3", \
     "--log-level", "INFO"]
