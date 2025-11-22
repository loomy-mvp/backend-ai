FROM python:3.11-slim

WORKDIR /app

COPY kb/scraping/ccia/ccia-modena/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY kb/scraping/ccia/ccia-modena/src/ ./src/
COPY kb/scraping/utils/ ./src/utils/
COPY kb/scraping/ccia/ccia-modena/run_scraper.py ./

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "run_scraper.py"]
CMD ["--bucket-name", "loomy-public-documents", \
     "--base-folder", "ccia-modena/statuto-regolamenti", \
     "--log-level", "INFO"]
