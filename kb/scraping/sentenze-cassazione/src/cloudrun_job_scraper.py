"""
cloudrun_job_scraper.py
-----------------------
Cloud Run Job for scraping Italian Supreme Court decisions.
Designed to run in parallel across multiple regions and years.

Usage as Cloud Run Job:
    gcloud run jobs execute italgiure-scraper-2024 \
        --region europe-west8 \
        --args="--year=2024,--sections=1,3,5,L"

Environment Variables:
    - GCP_SERVICE_ACCOUNT_CREDENTIALS: JSON credentials (optional on Cloud Run)
    - GCS_BUCKET: Bucket name (default: loomy-public-documents)
    - GCS_FOLDER: Folder path (default: sentenze-cassazione)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import random
import pickle
from collections import deque
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import requests
import urllib3
from google.cloud import storage
from google.oauth2 import service_account
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.upload_to_storage import upload_to_storage

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SOLR_ENDPOINT = (
    "https://www.italgiure.giustizia.it/sncass/isapi/hc.dll/"
    "sn.solr/sn-collection/select?app.query="
)

ATTACH_BASE = "https://www.italgiure.giustizia.it/xway/application/nif/clean/hc.dll"

COOKIES_BLOB_NAME = "sentenze-cassazione/italgiure_session_cookies.pkl"


def _load_cookies(bucket: storage.Bucket) -> Optional[requests.cookies.RequestsCookieJar]:
    """Load persisted cookies from GCS."""
    blob = bucket.blob(COOKIES_BLOB_NAME)
    if not blob.exists():
        return None
    try:
        data = blob.download_as_bytes()
        return pickle.loads(data)
    except Exception as e:
        print(f"⚠️  Failed to load cookies: {e}")
        return None


def _save_cookies(bucket: storage.Bucket, cookiejar: requests.cookies.RequestsCookieJar) -> None:
    """Persist cookies to GCS."""
    try:
        blob = bucket.blob(COOKIES_BLOB_NAME)
        data = pickle.dumps(cookiejar)
        blob.upload_from_string(data)
        print(f"💾 Saved session cookies to gs://{bucket.name}/{COOKIES_BLOB_NAME}")
    except Exception as e:
        print(f"⚠️  Failed to save cookies: {e}")



def _normalize_tokens(tokens: Optional[Iterable[str]]) -> Tuple[str, ...]:
    if not tokens:
        return tuple()
    # Preserve deterministic order for queue naming
    return tuple({token.strip(): token.strip() for token in tokens if token}.values())


def _queue_blob_path(
    folder: str,
    kind: Tuple[str, ...],
    sections: Tuple[str, ...],
    year: Optional[int],
    start_offset: int,
    max_documents: Optional[int],
) -> str:
    """Build the GCS object path holding the queue state for the run."""
    kind_key = "-".join(kind) if kind else "all"
    sections_key = "-".join(sections) if sections else "all"
    year_key = str(year) if year is not None else "all-years"
    offset_key = f"offset-{start_offset}"
    max_key = f"max-{max_documents}" if max_documents is not None else "max-all"
    queue_root = f"{folder}_queue_state"
    return f"{queue_root}/{year_key}/{kind_key}/{sections_key}/{offset_key}-{max_key}.json"


def _download_queue(bucket: storage.Bucket, blob_path: str) -> Optional[Dict[str, object]]:
    blob = bucket.blob(blob_path)
    if not blob.exists():
        return None
    raw = blob.download_as_text()
    return json.loads(raw)


def _save_queue(bucket: storage.Bucket, blob_path: str, queue: Dict[str, object]) -> None:
    blob = bucket.blob(blob_path)
    payload = json.dumps(queue, indent=2, ensure_ascii=False)
    blob.upload_from_string(payload, content_type="application/json")


def _delete_queue(bucket: storage.Bucket, blob_path: str) -> None:
    blob = bucket.blob(blob_path)
    if blob.exists():
        blob.delete()


def _flatten_kind(kind_value: object) -> Optional[str]:
    if isinstance(kind_value, list) and kind_value:
        return str(kind_value[0])
    if isinstance(kind_value, str):
        return kind_value
    return None


def _ensure_queue(
    bucket: storage.Bucket,
    blob_path: str,
    kind: Tuple[str, ...],
    sections: Tuple[str, ...],
    year: int,
    batch_size: int,
    start_offset: int,
    max_documents: Optional[int],
) -> Dict[str, object]:
    """Create or load the queue of pending documents for the given filters."""
    filters = {
        "kind": list(kind),
        "sections": list(sections),
        "year": year,
        "batch_size": batch_size,
        "start_offset": start_offset,
        "max_documents": max_documents,
    }

    existing = _download_queue(bucket, blob_path)
    if existing and existing.get("filters") == filters:
        return existing

    print("🔁 Building fresh queue state from SOLR search...")
    base_response = query_decisions(kind=kind, szdec=sections, year=year, rows=1, start=0)
    total_docs = int(base_response.get("numFound", 0))
    print(f"📊 SOLR reported {total_docs} documents for the selected filters")

    tasks: List[Dict[str, object]] = []
    for start in range(0, total_docs, batch_size):
        response = query_decisions(kind=kind, szdec=sections, year=year, rows=batch_size, start=start)
        docs = response.get("docs", [])
        if not docs:
            break
        for doc in docs:
            url = build_pdf_url(doc)
            filename = url.split("id=")[-1].split("/")[-1]
            sezione = extract_sezione_from_filename(filename)
            doc_year = extract_year_from_filename(filename)
            doc_kind = _flatten_kind(doc.get("kind"))
            tasks.append({
                "id": doc.get("id"),
                "url": url,
                "filename": filename,
                "sezione": sezione,
                "kind": doc_kind,
                "year": doc_year,
            })

    print(f"🗂️  Collected {len(tasks)} documents into the queue before sharding filters")

    if start_offset:
        tasks = tasks[start_offset:]
        print(f"⚙️  Applied start offset {start_offset}, {len(tasks)} tasks remain")

    if max_documents is not None:
        tasks = tasks[:max_documents]
        print(f"⚙️  Applied max documents {max_documents}, {len(tasks)} tasks remain")

    queue_data = {
        "version": 1,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "filters": filters,
        "source_total": total_docs,
        "pending": tasks,
        "original_pending_count": len(tasks),
        "processed_success": 0,
        "processed_skipped": 0,
    }

    _save_queue(bucket, blob_path, queue_data)
    print(f"✅ Queue stored at gs://{bucket.name}/{blob_path}")
    return queue_data


def _object_exists(bucket: storage.Bucket, object_path: str) -> bool:
    return bucket.blob(object_path).exists()


def query_decisions(
    kind: Iterable[str] = ("snciv",),
    szdec: Optional[Iterable[str]] = None,
    year: Optional[int] = None,
    rows: int = 10,
    start: int = 0,
) -> Dict[str, object]:
    """Query decisions from Italgiure."""
    # Build query
    kind_clause = " OR ".join([f'kind:"{k}"' for k in kind])
    clauses = [f"({kind_clause})"]
    
    if szdec:
        szdec_clause = " OR ".join([f'szdec:"{s}"' for s in szdec])
        clauses.append(f"({szdec_clause})")
    
    if year:
        clauses.append(f'anno:"{year}"')
    
    q = " AND ".join(clauses)
    
    payload = {
        "start": str(start),
        "rows": str(rows),
        "q": q,
        "wt": "json",
        "indent": "off",
        "sort": "pd desc,numdec desc",
        "fl": "id,filename,kind",
    }
    
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://www.italgiure.giustizia.it/",
        "Connection": "keep-alive",
        "DNT": "1"
    }
    
    resp = requests.post(SOLR_ENDPOINT, headers=headers, data=payload, verify=False, timeout=30)
    resp.raise_for_status()
    return resp.json().get("response", {})


def build_pdf_url(doc: Dict[str, object]) -> str:
    """Build PDF URL from document."""
    kind = doc.get("kind")
    if isinstance(kind, list):
        kind = kind[0]
    
    filename = doc.get("filename")
    if isinstance(filename, list):
        filename = filename[0]
    
    if not isinstance(filename, str):
        raise ValueError("Document missing filename")
    
    if not filename.endswith(".clean.pdf"):
        filename = filename.replace(".pdf", ".clean.pdf")
    
    return f"{ATTACH_BASE}?verbo=attach&db={kind}&id={filename}"


def extract_sezione_from_filename(filename: str) -> Optional[str]:
    """
    Extract sezione from filename.
    Example: snciv@s10@a2025@n27609@tD.clean.pdf -> s1
             snciv@sL0@a2025@n12345@tD.clean.pdf -> sL
    Pattern: @s{section}0@ where section is 1, 3, 5, or L
    
    Returns sezione code (e.g., 's1', 's3', 's5', 'sL') or None if not found.
    """
    import re
    # Pattern to match @s{code}0@ where code is 1, 3, 5, or L
    match = re.search(r'@s([135L])0@', filename)
    if match:
        return f's{match.group(1)}'
    return None


def extract_year_from_filename(filename: str) -> Optional[int]:
    """
    Extract year from filename.
    Example: snciv@s30@a2025@n28197@tO.clean.pdf -> 2025
    Pattern: @a{year}@ where year is a 4-digit number
    
    Returns year as int or None if not found.
    """
    import re
    match = re.search(r'@a(\d{4})@', filename)
    if match:
        return int(match.group(1))
    return None


def try_download_pdf(
    url: str,
    session: requests.Session,
    max_retries: int = 3
) -> Optional[bytes]:
    """
    Try to download PDF with retries using the provided session.
    Returns PDF bytes if successful, None if captcha detected.
    """
    for attempt in range(max_retries):
        try:
            # Add delay before request (except first attempt)
            if attempt > 0:
                delay = random.uniform(3, 7) * (attempt + 1)
                time.sleep(delay)
            
            # Make request
            resp = session.get(url, verify=False, timeout=30, allow_redirects=True)
            
            # Check for captcha/block indicators
            text_lower = resp.text.lower() if hasattr(resp, 'text') else ""
            if (resp.status_code in (403, 429) or 
                "captcha" in text_lower or 
                "recaptcha" in text_lower or 
                "h-captcha" in text_lower):
                print(f"  ⚠️  Captcha/block detected (attempt {attempt + 1}/{max_retries})")
                return None
            
            resp.raise_for_status()
            
            # Check if we got HTML instead of PDF
            content_type = resp.headers.get('content-type', '').lower()
            
            if 'html' in content_type or b'<!DOCTYPE html>' in resp.content[:100]:
                print(f"  ⚠️  HTML response (attempt {attempt + 1}/{max_retries})")
                return None
            
            if 'pdf' in content_type or b'%PDF' in resp.content[:10]:
                return resp.content
            
            print(f"  ⚠️  Unexpected content type: {content_type}")
            
        except Exception as e:
            print(f"  ❌ Error (attempt {attempt + 1}/{max_retries}): {e}")
            continue
    
    return None


def scrape_year(
    year: Optional[int],
    kind: Iterable[str] = ("snciv",),
    szdec: Optional[Iterable[str]] = None,
    bucket_name: str = "loomy-public-documents",
    folder: str = "sentenze-cassazione",
    batch_size: int = 100,
    min_delay: float = 2.0,
    max_delay: float = 4.0,
    start_offset: int = 0,
    max_documents: Optional[int] = None,
    reset_queue: bool = False,
) -> Dict[str, object]:
    """Process queued decisions for a specific year until the first captcha appears."""
    year_label = str(year) if year is not None else "ALL"

    print(f"\n{'='*80}")
    print(f"🚀 Starting scrape for year {year_label}")
    print(f"   Sections: {', '.join(szdec) if szdec else 'ALL'}")
    print(f"   Region: {os.getenv('CLOUD_RUN_REGION', 'unknown')}")
    print(f"   Start offset: {start_offset}")
    if max_documents:
        print(f"   Max documents: {max_documents}")
    print(f"{'='*80}\n")

    # Initialize GCS client, preferring injected credentials but falling back to ADC
    storage_client: storage.Client
    gcp_credentials_info = os.getenv("GCP_SERVICE_ACCOUNT_CREDENTIALS")

    if gcp_credentials_info:
        try:
            credentials_dict = json.loads(gcp_credentials_info)
            credentials = service_account.Credentials.from_service_account_info(credentials_dict)
            storage_client = storage.Client(credentials=credentials)
            print("🔐 Using service account credentials from environment variable")
        except Exception as cred_error:
            print(f"⚠️  Failed to parse provided service account credentials: {cred_error}")
            print("   Falling back to Application Default Credentials")
            storage_client = storage.Client()
    else:
        storage_client = storage.Client()
        print("🔐 Using Application Default Credentials")

    bucket = storage_client.bucket(bucket_name)

    # Load persisted session cookies
    cookies = _load_cookies(bucket)
    session = requests.Session()
    if cookies:
        session.cookies.update(cookies)
        print("🍪 Loaded persisted session cookies")
    
    # Set realistic headers
    session.headers.update({
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/126.0.0.0 Safari/537.36'
        ),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7',
        'Referer': 'https://www.italgiure.giustizia.it/',
        'Connection': 'keep-alive',
        'DNT': '1',
    })

    normalized_sections = _normalize_tokens(szdec)
    normalized_kind = _normalize_tokens(kind)
    queue_blob_path = _queue_blob_path(
        folder=folder,
        kind=normalized_kind,
        sections=normalized_sections,
        year=year,
        start_offset=start_offset,
        max_documents=max_documents,
    )

    if reset_queue:
        print(f"🧹 Reset requested. Deleting existing queue state at gs://{bucket.name}/{queue_blob_path} if present.")
        _delete_queue(bucket, queue_blob_path)

    queue_state = _ensure_queue(
        bucket=bucket,
        blob_path=queue_blob_path,
        kind=normalized_kind,
        sections=normalized_sections,
        year=year,
        batch_size=batch_size,
        start_offset=start_offset,
        max_documents=max_documents,
    )

    pending_tasks = deque(queue_state.get("pending", []))
    total_before_run = len(pending_tasks)

    if total_before_run == 0:
        print("ℹ️  Queue is empty. Nothing to process.")
        _delete_queue(bucket, queue_blob_path)
        return {
            "pending_before": 0,
            "pending_after": 0,
            "successful": 0,
            "skipped_existing": 0,
            "failed": 0,
            "captcha": 0,
            "attempted": 0,
        }

    max_attempts = total_before_run if max_documents is None else min(total_before_run, max_documents)

    stats = {
        "pending_before": total_before_run,
        "pending_after": total_before_run,
        "successful": 0,
        "skipped_existing": 0,
        "failed": 0,
        "captcha": 0,
        "attempted": 0,
    }

    start_time = time.time()
    processed_this_run = 0
    captcha_triggered = False

    while pending_tasks and processed_this_run < max_attempts:
        task = pending_tasks[0]
        filename = task.get("filename")
        url = task.get("url")
        sezione = task.get("sezione") or "unknown"
        doc_year = task.get("year")  # Year extracted from filename

        # Build folder: sentenze-cassazione/{year}/{sezione}
        if doc_year is not None:
            upload_folder = f"{folder}/{doc_year}/{sezione}"
        else:
            # Fallback if year not extracted from filename
            upload_folder = f"{folder}/{sezione}"
        destination_blob = f"{upload_folder}/{filename}"

        if _object_exists(bucket, destination_blob):
            print(f"✅ Already present: {destination_blob}. Removing from queue.")
            pending_tasks.popleft()
            processed_this_run += 1
            stats["skipped_existing"] += 1
            queue_state["processed_skipped"] = queue_state.get("processed_skipped", 0) + 1
            queue_state["pending"] = list(pending_tasks)
            queue_state["updated_at"] = datetime.utcnow().isoformat() + "Z"
            _save_queue(bucket, queue_blob_path, queue_state)
            continue

        print(f"[{processed_this_run + 1}/{max_attempts}] Downloading {filename}")
        pdf_bytes = try_download_pdf(url, session, max_retries=3)
        stats["attempted"] += 1

        if not pdf_bytes:
            print("⚠️  Captcha encountered. Persisting session cookies and stopping run.")
            _save_cookies(bucket, session.cookies)
            stats["captcha"] += 1
            captcha_triggered = True
            break

        try:
            pdf_obj = {
                "name": filename,
                "bytes": pdf_bytes,
                "extension": "pdf",
            }
            upload_to_storage(storage_client, bucket_name, pdf_obj, folder=upload_folder)
            stats["successful"] += 1
            queue_state["processed_success"] = queue_state.get("processed_success", 0) + 1
            print(f"  ✅ Uploaded to gs://{bucket_name}/{upload_folder}")
            
            # Save cookies after each successful download to keep session alive
            _save_cookies(bucket, session.cookies)
            
            pending_tasks.popleft()
            processed_this_run += 1
            queue_state["pending"] = list(pending_tasks)
            queue_state["updated_at"] = datetime.utcnow().isoformat() + "Z"
            _save_queue(bucket, queue_blob_path, queue_state)
        except Exception as err:
            stats["failed"] += 1
            print(f"  ❌ Upload failed: {err}")
            # Keep task in queue for next run
            break

        # Delay to stay polite even if we plan to stop at captcha
        if pending_tasks and processed_this_run < max_attempts:
            delay = random.uniform(min_delay, max_delay)
            time.sleep(delay)

    queue_state["pending"] = list(pending_tasks)
    queue_state["updated_at"] = datetime.utcnow().isoformat() + "Z"
    _save_queue(bucket, queue_blob_path, queue_state)

    stats["pending_after"] = len(pending_tasks)

    if not pending_tasks:
        print(f"🎯 Queue completed. Deleting gs://{bucket.name}/{queue_blob_path}.")
        _delete_queue(bucket, queue_blob_path)

    elapsed_time = time.time() - start_time
    processed_count = stats["successful"] + stats["skipped_existing"]

    print(f"\n📊 Run statistics:")
    print(f"   Pending before run: {stats['pending_before']}")
    print(f"   Pending after run: {stats['pending_after']}")
    print(f"   ✅ Uploaded this run: {stats['successful']}")
    print(f"   ♻️  Skipped (already uploaded): {stats['skipped_existing']}")
    print(f"   ⚠️  Captcha events: {stats['captcha']}")
    print(f"   ❌ Failures: {stats['failed']}")
    print(f"   ⏱️  Duration: {elapsed_time/60:.1f} minutes")

    if captcha_triggered:
        print("   🔁 Run stopped early due to captcha. Resume with next execution.")

    print(f"{'='*80}\n")

    stats["elapsed_seconds"] = elapsed_time
    stats["processed_count"] = processed_count
    stats["captcha_triggered"] = captcha_triggered

    return stats


def main():
    """Main entry point for Cloud Run Job."""
    parser = argparse.ArgumentParser(description='Scrape Italian Supreme Court decisions')
    parser.add_argument('--year', type=int, required=False, default=None,
                       help='Year to scrape (omit to include all available years)')
    parser.add_argument(
        '--sections',
        nargs='*',
        default=None,
        help='Section codes (space or comma separated). Default: 1 3 5 L'
    )
    parser.add_argument('--kind', type=str, default='snciv', 
                       help='Document kind: snciv or snpen (default: snciv)')
    parser.add_argument('--bucket', type=str, default='loomy-public-documents',
                       help='GCS bucket name')
    parser.add_argument('--folder', type=str, default='sentenze-cassazione',
                       help='GCS folder path')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Documents per batch (default: 100)')
    parser.add_argument('--min-delay', type=float, default=2.0,
                       help='Minimum delay between downloads in seconds')
    parser.add_argument('--max-delay', type=float, default=4.0,
                       help='Maximum delay between downloads in seconds')
    parser.add_argument('--start-offset', type=int, default=0,
                       help='Start index for sharded runs (default: 0)')
    parser.add_argument('--max-documents', type=int, default=None,
                       help='Maximum documents to process in this run')
    parser.add_argument('--reset-queue', action='store_true',
                       help='Drop any existing queue snapshot before starting')
    
    args = parser.parse_args()
    
    # Parse sections (support both comma-separated and space-separated input)
    if args.sections:
        raw_sections = []
        for token in args.sections:
            raw_sections.extend(part.strip() for part in token.split(',') if part.strip())
        sections = tuple(raw_sections)
    else:
        sections = ('1', '3', '5', 'L')
    
    # Parse kind
    kind = tuple(args.kind.split(','))
    
    try:
        stats = scrape_year(
            year=args.year,
            kind=kind,
            szdec=sections,
            bucket_name=args.bucket,
            folder=args.folder,
            batch_size=args.batch_size,
            min_delay=args.min_delay,
            max_delay=args.max_delay,
            start_offset=args.start_offset,
            max_documents=args.max_documents,
            reset_queue=args.reset_queue,
        )

        # Exit with success
        attempted = stats.get('attempted', 0)
        pending_after = stats.get('pending_after', 'unknown')
        success_rate = (stats['successful'] / attempted * 100) if attempted else 0.0
        year_desc = str(args.year) if args.year is not None else "all years"
        print(f"✅ Job completed successfully for {year_desc}")
        print(f"📊 Attempted downloads this run: {attempted}")
        print(f"   ✅ Successful uploads: {stats['successful']}")
        print(f"   ♻️  Skipped (already uploaded): {stats['skipped_existing']}")
        print(f"   ⚠️  Captcha triggered: {stats['captcha'] > 0}")
        print(f"   📦 Remaining in queue: {pending_after}")
        print(f"   🎯 Success rate (attempted only): {success_rate:.1f}%")
        sys.exit(0)

    except Exception as e:
        year_desc = str(args.year) if args.year is not None else "all years"
        print(f"❌ Job failed for {year_desc}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
