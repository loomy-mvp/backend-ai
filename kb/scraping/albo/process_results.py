"""
Post-processing utility for commercialisti scraper results
Includes data cleaning, deduplication, format conversion, and analysis
"""

import json
import csv
from typing import List, Dict, Optional, Set
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('commercially_processor.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class CommercialisitiDataProcessor:
    """Process and analyze scraped commercialisti data"""
    
    def __init__(self, json_file: str):
        """
        Initialize processor
        
        Args:
            json_file: Path to JSON file with scraper results
        """
        self.json_file = json_file
        self.records = []
        self.metadata = {}
        self.load_data()
        
    def load_data(self):
        """Load data from JSON file"""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, dict) and 'records' in data:
                self.metadata = data.get('metadata', {})
                self.records = data.get('records', [])
            else:
                self.records = data if isinstance(data, list) else []
                
            logger.info(f"[OK] Loaded {len(self.records)} records from {self.json_file}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            
    def get_stats(self) -> Dict:
        """
        Get basic statistics about the data
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_records': len(self.records),
            'unique_caps': len(set(r.get('cap_code', '') for r in self.records if r.get('cap_code'))),
            'unique_names': len(set(r.get('nome_completo', '') for r in self.records if r.get('nome_completo'))),
            'unique_ordini': len(set(r.get('ordine', '') for r in self.records if r.get('ordine'))),
            'records_with_sede': len([r for r in self.records if r.get('sede_studio')]),
            'revisori_contabili': len([r for r in self.records if r.get('revisore_contabile', '').upper() == 'SI']),
        }
        
        # Count by sezione
        sezioni = {}
        for record in self.records:
            sezione = record.get('sezione', 'Unknown')
            sezioni[sezione] = sezioni.get(sezione, 0) + 1
        stats['by_sezione'] = sezioni
        
        return stats
        
    def print_stats(self):
        """Print statistics"""
        stats = self.get_stats()
        
        logger.info(f"\n{'='*60}")
        logger.info("Data Statistics")
        logger.info(f"{'='*60}")
        logger.info(f"Total records: {stats['total_records']}")
        logger.info(f"Unique CAPs: {stats['unique_caps']}")
        logger.info(f"Unique names: {stats['unique_names']}")
        logger.info(f"Unique ordini: {stats['unique_ordini']}")
        logger.info(f"Records with office address: {stats['records_with_sede']}")
        logger.info(f"Certified auditors (SI): {stats['revisori_contabili']}")
        
        if stats['by_sezione']:
            logger.info(f"\nRecords by sezione:")
            for sezione, count in sorted(stats['by_sezione'].items()):
                logger.info(f"  {sezione}: {count}")
        
        logger.info(f"{'='*60}\n")
        
    def deduplicate(self, key_fields: Optional[List[str]] = None) -> List[Dict]:
        """
        Remove duplicate records
        
        Args:
            key_fields: Fields to use for deduplication (default: cap_code + nome_completo)
            
        Returns:
            Deduplicated records list
        """
        if key_fields is None:
            key_fields = ['cap_code', 'nome_completo']
            
        seen = set()
        deduped = []
        duplicates_removed = 0
        
        for record in self.records:
            key = tuple(record.get(field, '') for field in key_fields)
            if key not in seen and any(key):  # Skip empty keys
                seen.add(key)
                deduped.append(record)
            else:
                duplicates_removed += 1
                
        logger.info(f"[OK] Removed {duplicates_removed} duplicates")
        self.records = deduped
        return deduped
        
    def clean_data(self) -> int:
        """
        Clean data by standardizing formats and removing incomplete records
        
        Returns:
            Number of records removed
        """
        initial_count = len(self.records)
        cleaned = []
        
        for record in self.records:
            # Skip records with no identifying information
            if not record.get('nome_completo') and not record.get('ordine'):
                continue
                
            # Standardize date format (ensure DD/MM/YYYY)
            for date_field in ['data_nascita', 'data_anzianita', 'data_iscrizione', 'data_modifica']:
                if date_field in record and record[date_field]:
                    record[date_field] = self._standardize_date(record[date_field])
                    
            # Standardize yes/no fields
            if 'revisore_contabile' in record:
                record['revisore_contabile'] = record['revisore_contabile'].upper().strip()
                
            # Trim whitespace from all text fields
            for key, value in record.items():
                if isinstance(value, str):
                    record[key] = value.strip()
                    
            cleaned.append(record)
            
        records_removed = initial_count - len(cleaned)
        logger.info(f"[OK] Cleaned data: removed {records_removed} incomplete records")
        self.records = cleaned
        return records_removed
        
    def _standardize_date(self, date_str: str) -> str:
        """
        Standardize date format to DD/MM/YYYY
        
        Args:
            date_str: Date string in various formats
            
        Returns:
            Standardized date string
        """
        if not date_str:
            return ''
            
        date_str = date_str.strip()
        
        # Already in DD/MM/YYYY format
        if len(date_str) == 10 and date_str.count('/') == 2:
            return date_str
            
        # Try to parse and reformat
        try:
            from datetime import datetime
            # Try various formats
            for fmt in ['%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%d.%m.%Y']:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime('%d/%m/%Y')
                except ValueError:
                    continue
        except:
            pass
            
        return date_str
        
    def export_by_ordine(self, output_dir: str = "exports"):
        """
        Export records grouped by ordine (professional order)
        
        Args:
            output_dir: Directory to save grouped files
        """
        try:
            Path(output_dir).mkdir(exist_ok=True)
            
            # Group by ordine
            by_ordine = {}
            for record in self.records:
                ordine = record.get('ordine', 'Unknown')
                if ordine not in by_ordine:
                    by_ordine[ordine] = []
                by_ordine[ordine].append(record)
                
            # Export each ordine
            for ordine, records in by_ordine.items():
                safe_filename = "".join(c for c in ordine if c.isalnum() or c in " -_").strip()
                safe_filename = safe_filename.replace(" ", "_")[:50]
                csv_file = f"{output_dir}/commercialisti_{safe_filename}.csv"
                
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    if records:
                        fieldnames = list(records[0].keys())
                        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
                        writer.writeheader()
                        writer.writerows(records)
                        
                logger.info(f"  [OK] {ordine}: {len(records)} records -> {csv_file}")
                
            logger.info(f"[OK] Exported records grouped by ordine to {output_dir}/")
            
        except Exception as e:
            logger.error(f"Error exporting by ordine: {e}")
            
    def export_json(self, output_file: Optional[str] = None):
        """
        Export records to JSON
        
        Args:
            output_file: Output filename
        """
        if not output_file:
            output_file = self.json_file.replace('.json', '_processed.json')
            
        try:
            data = {
                'metadata': {
                    'total_records': len(self.records),
                    'processed_at': datetime.now().isoformat(),
                    'source': 'https://ricerca.commercialisti.it/RicercaIscritti'
                },
                'records': self.records
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"[OK] Exported to JSON: {output_file}")
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            
    def export_csv(self, output_file: Optional[str] = None):
        """
        Export records to CSV
        
        Args:
            output_file: Output filename
        """
        if not output_file:
            output_file = self.json_file.replace('.json', '_processed.csv')
            
        try:
            if not self.records:
                logger.warning("No records to export")
                return
                
            fieldnames = list(self.records[0].keys())
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
                writer.writeheader()
                writer.writerows(self.records)
                
            logger.info(f"[OK] Exported to CSV: {output_file}")
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")


def main():
    """Main entry point"""
    import sys
    
    json_file = "commercialisti_data_v2.json"
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        
    logger.info(f"\nProcessing: {json_file}\n")
    
    processor = CommercialisitiDataProcessor(json_file)
    
    # Print statistics
    processor.print_stats()
    
    # Clean data
    processor.clean_data()
    
    # Deduplicate
    processor.deduplicate()
    
    # Print cleaned statistics
    logger.info("After cleaning:")
    processor.print_stats()
    
    # Export processed results
    processor.export_json()
    processor.export_csv()
    processor.export_by_ordine()
    
    logger.info(f"[OK] Processing complete!")


if __name__ == "__main__":
    main()
