# Dockerfile for Cloud Run Job - INPS Scraper
FROM python:3.11-slim

# Install Chrome dependencies and Chrome
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    curl \
    && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome-keyring.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Install ChromeDriver matching Chrome version
RUN CHROME_VERSION=$(google-chrome --version | awk '{print $3}' | cut -d. -f1) \
    && wget -O /tmp/chrome-for-testing.json https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json \
    && CHROMEDRIVER_URL=$(python3 -c "import json, sys; data=json.load(open('/tmp/chrome-for-testing.json')); print(data['channels']['Stable']['downloads']['chromedriver'][0]['url'] if 'chromedriver' in data['channels']['Stable']['downloads'] else '')") \
    && if [ -z "$CHROMEDRIVER_URL" ]; then \
         CHROMEDRIVER_URL="https://storage.googleapis.com/chrome-for-testing-public/$(curl -s https://googlechromelabs.github.io/chrome-for-testing/LATEST_RELEASE_STABLE)/linux64/chromedriver-linux64.zip"; \
       fi \
    && wget -O /tmp/chromedriver.zip "$CHROMEDRIVER_URL" \
    && unzip /tmp/chromedriver.zip -d /tmp/ \
    && mv /tmp/chromedriver-linux64/chromedriver /usr/local/bin/ \
    && chmod +x /usr/local/bin/chromedriver \
    && rm -rf /tmp/chromedriver.zip /tmp/chromedriver-linux64 /tmp/chrome-for-testing.json

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY kb/scraping/inps/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the scraping module with correct paths
COPY kb/scraping/inps/src/ ./src/
COPY kb/scraping/utils/ ./src/utils/
COPY kb/scraping/inps/run_scraper.py .

# Set Python path
ENV PYTHONPATH=/app

# Use ENTRYPOINT to allow passing arguments
ENTRYPOINT ["python", "run_scraper.py"]

# Default arguments (can be overridden at runtime)
CMD ["--bucket-name", "loomy-public-documents", \
     "--base-folder", "inps", \
     "--headless", \
     "--max-pages", "1000", \
     "--log-level", "INFO"]
