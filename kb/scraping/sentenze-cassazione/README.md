# Sentenze Cassazione Scraper with Proxy Support

Scraper for Italian Supreme Court decisions (Corte di Cassazione) with built-in proxy rotation to bypass captcha protection.

## Features

✅ **Proxy Rotation** - Automatically rotates through multiple IP addresses  
✅ **Captcha Detection** - Detects and skips captcha-protected downloads  
✅ **Health Monitoring** - Tracks proxy success/failure rates  
✅ **User Agent Rotation** - Randomizes browser signatures  
✅ **Metadata Collection** - Saves all document metadata to JSON  
✅ **GCS Upload** - Automatically uploads PDFs to Google Cloud Storage  
✅ **Progress Tracking** - Real-time statistics and progress updates  

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Proxies

**Option A: Use Free Proxies (Testing Only)**
```bash
python src/fetch_free_proxies.py
```
This will test free proxies and save working ones to `src/proxies.txt`.

**Option B: Use Paid Proxies (Recommended)**

Add to your `.env` file:
```bash
PROXY_LIST="http://user:pass@proxy1.com:8080,http://user:pass@proxy2.com:8080"
```

Or create `src/proxies.txt`:
```
http://user:pass@proxy1.com:8080
http://user:pass@proxy2.com:8080
```

See [PROXY_SETUP.md](PROXY_SETUP.md) for detailed proxy service recommendations.

### 3. Configure Environment

Add to `.env`:
```bash
GCP_SERVICE_ACCOUNT_CREDENTIALS='{"type": "service_account", ...}'
PROXY_LIST="http://proxy1:port,http://proxy2:port"  # Optional
```

### 4. Run the Scraper

```bash
python src/scrape_italgiure_simple.py
```

## Configuration

Edit `src/scrape_italgiure_simple.py` main() function:

```python
scrape_with_delays(
    kind=("snciv",),              # Document type: snciv=civil, snpen=criminal
    szdec=("1", "3", "5", "L"),   # Sections: 1=PRIMA, 3=TERZA, 5=QUINTA, L=LAVORO
    year=2025,                     # Filter by year (or None for all)
    bucket_name="loomy-kb",        # GCS bucket
    folder="sentenze-cassazione",  # Folder in bucket
    batch_size=100,                # Documents per batch
    min_delay=2.0,                 # Min seconds between downloads
    max_delay=2.5,                 # Max seconds between downloads
    save_metadata=True,            # Save metadata JSON
    use_proxies=True               # Enable proxy rotation
)
```

## How It Works

1. **Query API** - Fetches document metadata from Italgiure SOLR API
2. **Build URLs** - Constructs PDF download URLs
3. **Rotate Proxies** - Uses next proxy from pool for each download
4. **Download PDF** - Attempts download with captcha detection
5. **Upload to GCS** - Uploads successful downloads to Google Cloud Storage
6. **Track Stats** - Monitors success rates and proxy health

## Proxy Rotation

The scraper includes smart proxy rotation:

- **Round-robin rotation** - Cycles through all available proxies
- **Health tracking** - Monitors success/failure per proxy
- **Auto-blacklisting** - Disables proxies after 5+ failures
- **Statistics** - Shows proxy performance after each batch

Example output:
```
📊 Proxy Statistics:
  ✅ ACTIVE http://proxy1.com:8080 - Success: 45, Failed: 2, Rate: 95.7%
  ✅ ACTIVE http://proxy2.com:8080 - Success: 38, Failed: 5, Rate: 88.4%
  ❌ FAILED http://proxy3.com:8080 - Success: 5, Failed: 12, Rate: 29.4%
```

## Output

### Console Output
```
🔍 Querying documents...
✅ Loaded 10 proxies from PROXY_LIST environment variable
🔄 Proxy rotation enabled with 10 proxies
📊 Found 15847 documents

📥 Batch 0 to 100 of 15847

[1/15847] snciv@s30@a2025@n27609@tD.clean.pdf
  🔄 Using proxy: http://proxy1.com:8080...
  ✅ Uploaded
  ⏳ Waiting 2.3s...

✅ Progress: 1 successful | ⚠️  0 captcha | ❌ 0 failed
```

### Metadata File
A `sentenze_metadata.json` file is created with all document information:
```json
[
  {
    "id": "...",
    "filename": "snciv@s30@a2025@n27609@tD.clean.pdf",
    "kind": "snciv",
    "url": "https://...",
  },
  ...
]
```

## Troubleshooting

### Getting Captcha Blocked?

1. **Add more proxies** - More IPs = less chance of blocks
2. **Increase delays** - Set `min_delay=5.0, max_delay=10.0`
3. **Use residential proxies** - More expensive but more reliable
4. **Check proxy quality** - Review proxy statistics output

### All Proxies Failing?

1. **Test proxies manually**:
   ```bash
   python src/fetch_free_proxies.py
   ```

2. **Verify proxy format**:
   ```
   http://username:password@host:port  ✅ Correct
   host:port                            ❌ Missing protocol
   ```

3. **Check credentials** - Ensure username/password are correct

### Slow Downloads?

- Use faster proxy service (datacenter > residential)
- Reduce `max_retries` in code
- Increase `batch_size` for better throughput

## Recommended Proxy Services

| Service | Cost/Month | Type | Best For |
|---------|------------|------|----------|
| **Webshare** | $40 | Datacenter | Budget |
| **Smartproxy** | $75 | Residential | Balance |
| **Bright Data** | $500 | Residential | Premium |
| **ScraperAPI** | $49 | Managed | Easiest |

See [PROXY_SETUP.md](PROXY_SETUP.md) for detailed comparison.

## Cost Estimation

For ~100,000 documents:

- **Without proxies**: ~10% success rate (captcha blocks most)
- **With 10 datacenter proxies**: ~60% success rate
- **With 50 residential proxies**: ~90% success rate
- **With rotating residential**: ~95% success rate

## Files

```
sentenze-cassazione/
├── src/
│   ├── scrape_italgiure_simple.py    # Main scraper with proxy support
│   ├── fetch_free_proxies.py         # Fetch and test free proxies
│   └── proxies.txt                   # Proxy list (create this)
├── PROXY_SETUP.md                    # Detailed proxy setup guide
├── README.md                         # This file
└── requirements.txt                  # Python dependencies
```

## Notes

- **Rate Limiting**: Even with proxies, use reasonable delays (2-5 seconds)
- **Metadata**: Always saved to JSON - useful for failed downloads
- **Resuming**: Check GCS before downloading (skips existing files)
- **Legal**: Ensure scraping complies with website terms of service

## Support

For issues or questions:
1. Check [PROXY_SETUP.md](PROXY_SETUP.md) for proxy configuration
2. Review proxy statistics output for failed proxies
3. Test proxies manually with `fetch_free_proxies.py`
4. Consider using paid proxy service for better results
