"""
Scraper for Italian CAP (postal codes) from comuni-italiani.it
Scrapes pages from 00.html to 97.html and extracts CAP codes with their municipalities
"""

import requests
from bs4 import BeautifulSoup
import csv
import time
from typing import List, Tuple
import re

def scrape_cap_page(page_num: str) -> List[Tuple[str, str]]:
    """
    Scrape a single CAP page and extract CAP code - comune pairs
    
    Args:
        page_num: Two-digit page number (e.g., '00', '01', '97')
    
    Returns:
        List of tuples (cap_code, comune)
    """
    url = f"https://www.comuni-italiani.it/cap/{page_num}.html"
    print(f"Scraping {url}...")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return []
    
    soup = BeautifulSoup(response.content, 'html.parser')
    cap_data = []
    
    # Find all tables in the page
    tables = soup.find_all('table')
    
    for table in tables:
        rows = table.find_all('tr')
        
        for row in rows:
            cells = row.find_all('td')
            
            # Each row has 4 cells: CAP | Comune | CAP | Comune
            # Process in pairs
            if len(cells) >= 2:
                # First pair
                cap1 = cells[0].get_text(strip=True)
                comune1 = cells[1].get_text(strip=True)
                
                # Only add if both CAP and comune are valid
                # CAP must be exactly 5 digits or match pattern like 001xx
                if cap1 and comune1 and cap1 != '---' and comune1:
                    # Skip header rows and invalid entries
                    # CAP must be exactly 5 characters (digits or 'x') and comune must not contain digits
                    if (not cap1.startswith('CAP') and 
                        re.match(r'^(\d{5}|\d{3}xx)$', cap1) and 
                        len(cap1) == 5 and
                        not re.search(r'\d{5}', comune1)):  # Comune shouldn't contain CAP codes
                        cap_data.append((cap1, comune1))
                
                # Second pair (if exists)
                if len(cells) >= 4:
                    cap2 = cells[2].get_text(strip=True)
                    comune2 = cells[3].get_text(strip=True)
                    
                    if cap2 and comune2 and cap2 != '---' and comune2:
                        if (not cap2.startswith('CAP') and 
                            re.match(r'^(\d{5}|\d{3}xx)$', cap2) and 
                            len(cap2) == 5 and
                            not re.search(r'\d{5}', comune2)):
                            cap_data.append((cap2, comune2))
    
    print(f"  Found {len(cap_data)} CAP entries")
    return cap_data


def scrape_all_caps() -> List[Tuple[str, str]]:
    """
    Scrape all CAP pages from 00.html to 97.html
    
    Returns:
        List of all (cap_code, comune) tuples
    """
    all_caps = []
    
    # Iterate through pages 00 to 97
    for i in range(98):
        page_num = f"{i:02d}"  # Format as two digits with leading zero
        cap_data = scrape_cap_page(page_num)
        all_caps.extend(cap_data)
        
        # Be polite to the server - add a small delay between requests
        time.sleep(0.5)
    
    return all_caps


def save_to_csv(data: List[Tuple[str, str]], filename: str):
    """
    Save CAP data to CSV file
    
    Args:
        data: List of (cap_code, comune) tuples
        filename: Output CSV filename
    """
    print(f"\nSaving {len(data)} entries to {filename}...")
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['cap_code', 'comune'])  # Header
        writer.writerows(data)
    
    print(f"Successfully saved to {filename}")


def main():
    """Main execution function"""
    print("Starting CAP scraper...")
    print("=" * 60)
    
    # Scrape all CAP codes
    all_caps = scrape_all_caps()
    
    print("=" * 60)
    print(f"Total CAP entries scraped: {len(all_caps)}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_caps = []
    for cap, comune in all_caps:
        key = (cap, comune)
        if key not in seen:
            seen.add(key)
            unique_caps.append((cap, comune))
    
    print(f"Unique CAP entries: {len(unique_caps)}")
    
    # Save to CSV
    output_file = "cap_comuni.csv"
    save_to_csv(unique_caps, output_file)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
