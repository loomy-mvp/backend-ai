# Dockerfile for Cloud Run Job - ODCEC Modena Scraper
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
RUN CHROME_VERSION=$(google-chrome --version | awk '{print $3}') \
    && CHROME_MAJOR=$(echo $CHROME_VERSION | cut -d. -f1) \
    && echo "Chrome version: $CHROME_VERSION (major: $CHROME_MAJOR)" \
    && wget -O /tmp/chrome-for-testing.json "https://googlechromelabs.github.io/chrome-for-testing/known-good-versions-with-downloads.json" \
    && CHROMEDRIVER_URL=$(python3 -c "import json; data=json.load(open('/tmp/chrome-for-testing.json')); versions=[v for v in data['versions'] if v['version'].startswith('$CHROME_MAJOR.')]; latest=sorted(versions, key=lambda x: x['version'], reverse=True)[0] if versions else None; print(latest['downloads']['chromedriver'][0]['url'] if latest and 'chromedriver' in latest['downloads'] else '')") \
    && if [ -z "$CHROMEDRIVER_URL" ]; then \
         echo "Falling back to LATEST_RELEASE for major version $CHROME_MAJOR"; \
         LATEST_VERSION=$(curl -s "https://googlechromelabs.github.io/chrome-for-testing/LATEST_RELEASE_$CHROME_MAJOR") && \
         CHROMEDRIVER_URL="https://storage.googleapis.com/chrome-for-testing-public/$LATEST_VERSION/linux64/chromedriver-linux64.zip"; \
       fi \
    && echo "Downloading ChromeDriver from: $CHROMEDRIVER_URL" \
    && wget -O /tmp/chromedriver.zip "$CHROMEDRIVER_URL" \
    && unzip /tmp/chromedriver.zip -d /tmp/ \
    && mv /tmp/chromedriver-linux64/chromedriver /usr/local/bin/ \
    && chmod +x /usr/local/bin/chromedriver \
    && rm -rf /tmp/chromedriver.zip /tmp/chromedriver-linux64 /tmp/chrome-for-testing.json

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY kb/scraping/odcec-modena/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the scraping module with correct paths
COPY kb/scraping/odcec-modena/src/ ./src/
COPY kb/scraping/utils/ ./src/utils/
COPY kb/scraping/odcec-modena/run_scraper.py .

# Set Python path
ENV PYTHONPATH=/app

# Use ENTRYPOINT to allow passing arguments
ENTRYPOINT ["python", "run_scraper.py"]

# Default arguments (can be overridden at runtime)
CMD ["--bucket-name", "loomy-public-documents", \
     "--base-folder", "odcec-modena", \
     "--headless", \
     "--log-level", "INFO"]
