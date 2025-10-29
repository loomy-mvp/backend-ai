from ladle.ladle import Ladle
import requests
from datetime import datetime
import time
import os
import json
# Configuration for scraping
import urllib.parse
import urllib3
import sys
# Google Cloud Storage
from utils.upload_to_storage import upload_to_storage
from google.oauth2 import service_account
from google.cloud import storage
gcp_credentials_info = os.getenv("GCP_SERVICE_ACCOUNT_CREDENTIALS")
gcp_credentials_info = json.loads(gcp_credentials_info)
gcp_service_account_credentials = service_account.Credentials.from_service_account_info(gcp_credentials_info)
storage_client = storage.Client(credentials=gcp_service_account_credentials)

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import StaleElementReferenceException

ladle = Ladle(headless=False)
ladle.driver.get("https://www.italgiure.giustizia.it/sncass/")

anno_corrente = datetime.now().year
button_mapping = {
    "archivio": {
        "CIVILE": "/html/body/div[1]/div[3]/form/table/tbody[2]/tr/td[1]/table/tbody/tr[2]/td/div/div/div/div/div/table/tbody/tr[1]/td[2]",
        "PENALE": "/html/body/div[1]/div[3]/form/table/tbody[2]/tr/td[1]/table/tbody/tr[2]/td/div/div/div/div/div/table/tbody/tr[2]/td[2]"
    },
    "tipo": {
        "Decreto": "/html/body/div[1]/div[3]/form/table/tbody[2]/tr/td[1]/table/tbody/tr[3]/td/div/div/div/div/div/table/tbody/tr[1]/td[2]",
        "Ordinanza": "/html/body/div[1]/div[3]/form/table/tbody[2]/tr/td[1]/table/tbody/tr[3]/td/div/div/div/div/div/table/tbody/tr[2]/td[2]",
        "Ordinanza Interlocutoria": "/html/body/div[1]/div[3]/form/table/tbody[2]/tr/td[1]/table/tbody/tr[3]/td/div/div/div/div/div/table/tbody/tr[3]/td[2]",
        "Sentenza": "/html/body/div[1]/div[3]/form/table/tbody[2]/tr/td[1]/table/tbody/tr[3]/td/div/div/div/div/div/table/tbody/tr[4]/td[2]"
    },
    "sezione": {
        "PRIMA": "/html/body/div[1]/div[3]/form/table/tbody[2]/tr/td[1]/table/tbody/tr[4]/td/div/div/div/div/div/table/tbody/tr[1]/td[2]",
        "SECONDA": "/html/body/div[1]/div[3]/form/table/tbody[2]/tr/td[1]/table/tbody/tr[4]/td/div/div/div/div/div/table/tbody/tr[2]/td[2]",
        "TERZA": "/html/body/div[1]/div[3]/form/table/tbody[2]/tr/td[1]/table/tbody/tr[4]/td/div/div/div/div/div/table/tbody/tr[3]/td[2]",
        "QUARTA": "/html/body/div[1]/div[3]/form/table/tbody[2]/tr/td[1]/table/tbody/tr[4]/td/div/div/div/div/div/table/tbody/tr[4]/td[2]",
        "QUINTA": "/html/body/div[1]/div[3]/form/table/tbody[2]/tr/td[1]/table/tbody/tr[4]/td/div/div/div/div/div/table/tbody/tr[5]/td[2]",
        "SESTA": "/html/body/div[1]/div[3]/form/table/tbody[2]/tr/td[1]/table/tbody/tr[4]/td/div/div/div/div/div/table/tbody/tr[6]/td[2]",
        "SETTIMA": "/html/body/div[1]/div[3]/form/table/tbody[2]/tr/td[1]/table/tbody/tr[4]/td/div/div/div/div/div/table/tbody/tr[7]/td[2]",
        "FERIALE": "/html/body/div[1]/div[3]/form/table/tbody[2]/tr/td[1]/table/tbody/tr[4]/td/div/div/div/div/div/table/tbody/tr[8]/td[2]",
        "LAVORO": "/html/body/div[1]/div[3]/form/table/tbody[2]/tr/td[1]/table/tbody/tr[4]/td/div/div/div/div/div/table/tbody/tr[9]/td[2]",
        "UNITE": "/html/body/div[1]/div[3]/form/table/tbody[2]/tr/td[1]/table/tbody/tr[4]/td/div/div/div/div/div/table/tbody/tr[10]/td[2]",
    },
    "anno": { # Max 5 anni + corrente
        f"{anno_corrente}": "/html/body/div[1]/div[3]/form/table/tbody[2]/tr/td[1]/table/tbody/tr[5]/td/div/div/div/div/div/table/tbody/tr[1]/td[2]",
        f"{anno_corrente - 1}": "/html/body/div[1]/div[3]/form/table/tbody[2]/tr/td[1]/table/tbody/tr[5]/td/div/div/div/div/div/table/tbody/tr[2]/td[2]",
        f"{anno_corrente - 2}": "/html/body/div[1]/div[3]/form/table/tbody[2]/tr/td[1]/table/tbody/tr[5]/td/div/div/div/div/div/table/tbody/tr[3]/td[2]",
        f"{anno_corrente - 3}": "/html/body/div[1]/div[3]/form/table/tbody[2]/tr/td[1]/table/tbody/tr[5]/td/div/div/div/div/div/table/tbody/tr[4]/td[2]",
        f"{anno_corrente - 4}": "/html/body/div[1]/div[3]/form/table/tbody[2]/tr/td[1]/table/tbody/tr[5]/td/div/div/div/div/div/table/tbody/tr[5]/td[2]",
        f"{anno_corrente - 5}": "/html/body/div[1]/div[3]/form/table/tbody[2]/tr/td[1]/table/tbody/tr[5]/td/div/div/div/div/div/table/tbody/tr[6]/td[2]"
    }
}

time.sleep(3)
ladle.clicks.click(button_mapping['sezione']['PRIMA'])
time.sleep(3)
ladle.clicks.click(button_mapping['sezione']['TERZA'])
time.sleep(3)
ladle.clicks.click(button_mapping['sezione']['QUINTA'])
time.sleep(3)
ladle.clicks.click(button_mapping['sezione']['LAVORO'])
time.sleep(3)
# L'archivio va cliccato per ultimo perché la CIVILE elimina alcune sezioni e perciò riordina gli xpath e.g. td[5] -> td[4]
ladle.clicks.click(button_mapping['archivio']['CIVILE'])

# Disable SSL warnings (since we'll disable verification)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

main_folder = 'cassazione_sentenze'
bucket_name = "loomy-public-documents"
base_url = "https://www.italgiure.giustizia.it"  # Fixed: removed /sncass since data-arg contains full path

# Initialize results list
all_pdf_objs = []
page_counter = 0
max_pages = None  # Set to a number to limit pages for testing, or None for all pages

# Create a persistent session for efficient connection reuse
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
})
# Disable SSL verification for this site (Italian government sites sometimes have cert issues)
session.verify = False

# XPath for pagination button
new_page_button_xpath = "/html/body/div[1]/div[3]/form/table/tbody[2]/tr/td[2]/div/div[2]/div[11]/span"

# Function to check if pagination button exists and is clickable
def has_next_page():
    try:
        button = ladle.elements.element(new_page_button_xpath, timeout=2)
        is_displayed = ladle.elements.element_displayed(element=button)
        # Also check if the button is enabled (not disabled)
        if is_displayed:
            button_class = button.get_attribute('class') or ''
            # Button might be disabled with certain classes or aria attributes
            return 'disabled' not in button_class.lower()
        return False
    except Exception as e:
        print(f"No next page button found: {e}")
        return False

# Function to extract PDF URL from link element
def extract_pdf_url(link_element):
    """Extract the actual PDF URL from the link's data-arg attribute"""
    try:
        # Find the img tag inside the link
        img = link_element.find_element(By.TAG_NAME, 'img')
        data_arg = img.get_attribute('data-arg')
        
        if data_arg:
            # URL decode the path
            decoded_path = urllib.parse.unquote(data_arg)
            # Construct full URL
            full_url = f"{base_url}{decoded_path}"
            return full_url
        return None
    except Exception as e:
        print(f"    Error extracting PDF URL: {e}")
        return None

# Function to scrape current page
def scrape_current_page():
    """Scrape all PDF links from the current page"""
    scraped_count = 0
    try:
        # Get all PDF links on current page
        # The XPath pattern with [*] should match all result divs
        pdf_list = ladle.elements.elements('/html/body/div[1]/div[3]/form/table/tbody[2]/tr/td[2]/div/div[2]/div[*]/div/h3/a')
        
        print(f"Found {len(pdf_list)} documents on this page")
        
        for n in range(len(pdf_list)):
            try:
                # Re-fetch element to avoid stale reference
                doc_link = ladle.elements.element(f'/html/body/div[1]/div[3]/form/table/tbody[2]/tr/td[2]/div/div[2]/div[{n+1}]/div/h3/a', timeout=2)
                doc_text = doc_link.text
                
                print(f"  [{n+1}/{len(pdf_list)}] Processing: {doc_text[:80]}...")
                
                # Extract the actual PDF URL from the data-arg attribute
                doc_url = extract_pdf_url(doc_link)
                
                if not doc_url:
                    print(f"    ⚠️ Could not extract PDF URL")
                    continue
                
                # Download PDF content using session (with SSL verification disabled)
                try:
                    response = session.get(doc_url, timeout=60, verify=False)
                    response.raise_for_status()
                    pdf_bytes = response.content
                    
                    if pdf_bytes:
                        pdf_obj = {
                            "url": doc_url,
                            "name": doc_text,
                            "bytes": pdf_bytes,
                            "extension": "pdf"
                        }
                        
                        # Store in list
                        all_pdf_objs.append(pdf_obj)
                        
                        # Upload to Google Cloud Storage
                        upload_to_storage(
                            storage_client=storage_client,
                            bucket_name=bucket_name,
                            pdf_obj=pdf_obj,
                            folder=main_folder
                        )
                        scraped_count += 1
                        print(f"    ✅ Downloaded and uploaded ({len(pdf_bytes)/1024:.1f} KB)")
                    else:
                        print(f"    ⚠️ No content downloaded")
                        
                except requests.exceptions.Timeout:
                    print(f"    ⚠️ Timeout while downloading PDF")
                except requests.exceptions.RequestException as e:
                    print(f"    ⚠️ Request error: {e}")
                except Exception as e:
                    print(f"    ⚠️ Unexpected error: {e}")
                    
            except Exception as e:
                print(f"  ⚠️ Error processing document {n+1}: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Error scraping page: {e}")
    
    return scraped_count

# Main scraping loop
print("🚀 Starting Cassazione scraping...")
print("="*80)

try:
    while True:
        page_counter += 1
        print(f"\n📄 PAGE {page_counter}")
        print("-"*80)
        
        # Scrape current page
        scraped = scrape_current_page()
        print(f"✅ Scraped {scraped} PDFs from page {page_counter}")
        print(f"📊 Total PDFs collected so far: {len(all_pdf_objs)}")
        
        # Check if we should stop (max pages limit)
        if max_pages and page_counter >= max_pages:
            print(f"\n⏹️ Reached maximum page limit ({max_pages})")
            break
        
        # Check if there's a next page
        if not has_next_page():
            print(f"\n✅ No more pages available. Scraping complete!")
            break
        
        # Click next page button
        try:
            print(f"➡️ Navigating to page {page_counter + 1}...")
            ladle.clicks.click(new_page_button_xpath)
            time.sleep(2)  # Wait for page to load
        except Exception as e:
            print(f"❌ Failed to navigate to next page: {e}")
            break

finally:
    # Always close the session
    session.close()
    print("\n🔒 Session closed")

print("="*80)
print(f"🎉 SCRAPING COMPLETE!")
print(f"📊 Total pages processed: {page_counter}")
print(f"📚 Total PDFs collected: {len(all_pdf_objs)}")
print(f"☁️ All PDFs uploaded to gs://{bucket_name}/{main_folder}")