"""
Test script to process a document locally and see the complete flow:
1. Extract pages from PDF
2. Check for CID corruption (will skip if found)
3. Chunk the document
4. Show results without embedding

This tests the document processing pipeline without requiring API keys.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.utils.ai_workflow_utils.document_processing import get_document_processor, has_cid_corruption


def test_process_document(pdf_path: str):
    """Test the complete document processing flow locally."""
    
    print("=" * 80)
    print("DOCUMENT PROCESSING TEST")
    print("=" * 80)
    print(f"\n📄 Testing file: {pdf_path}\n")
    
    # Step 1: Read PDF file
    if not Path(pdf_path).exists():
        print(f"❌ Error: File not found: {pdf_path}")
        return
    
    with open(pdf_path, 'rb') as f:
        file_bytes = f.read()
    
    file_size_mb = len(file_bytes) / (1024 * 1024)
    print(f"✅ File loaded: {file_size_mb:.2f} MB")
    
    # Step 2: Process with document processor (like the real pipeline)
    print("\n" + "-" * 80)
    print("STAGE 1: Document Processing (checks for CID corruption)")
    print("-" * 80)
    
    try:
        from backend.services.kb_api import chunk_document
        
        doc_name = Path(pdf_path).stem
        processor = get_document_processor("application/pdf", pdf_path)
        
        # This will raise ValueError if CID corruption is detected
        chunks = processor.process(
            file_bytes,
            chunk_document=chunk_document,
            storage_path=pdf_path,
            doc_name=doc_name,
        )
        
        print(f"✅ Document processed successfully")
        print(f"✅ Generated {len(chunks)} chunks")
        
        # Analyze chunks
        if chunks:
            total_chars = sum(len(chunk["text"]) for chunk in chunks)
            avg_chunk_size = total_chars / len(chunks)
            
            print(f"\n📊 Chunk Statistics:")
            print(f"   Total chunks: {len(chunks)}")
            print(f"   Average chunk size: {avg_chunk_size:.0f} characters")
            print(f"   Total text: {total_chars:,} characters")
            
            # Show first 3 chunks
            print(f"\n📝 First 3 chunks:")
            for i, chunk in enumerate(chunks[:3], 1):
                print(f"\n   Chunk {i} (page {chunk['page']}):")
                print(f"   Length: {len(chunk['text'])} chars")
                preview = chunk["text"][:150].replace("\n", " ")
                print(f"   Text: \"{preview}...\"")
        
        # Summary
        print("\n" + "=" * 80)
        print("✅ TEST PASSED - Document is clean")
        print("=" * 80)
        print(f"\n📊 Summary:")
        print(f"  • Chunks created: {len(chunks)}")
        print(f"  • Average chunk size: {avg_chunk_size:.0f} chars")
        print(f"  • No CID corruption detected")
        
        return chunks
        
    except ValueError as e:
        # CID corruption detected - document will be skipped
        print(f"\n❌ Document skipped: {e}")
        print("\n" + "=" * 80)
        print("⚠️  TEST RESULT: Document Skipped (CID Corruption)")
        print("=" * 80)
        print(f"\n📊 Summary:")
        print(f"  • This PDF contains CID encoding corruption")
        print(f"  • It uses symbolic fonts that cannot be decoded")
        print(f"  • Document will be skipped during batch processing")
        print(f"  • Solution: Requires OCR or manual conversion")
        return None
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_process_document.py <path_to_pdf>")
        print("\nExample:")
        print("  python test_process_document.py \"kb/old/circolari_2003_Circolare n.10 del'11 dicembre 2003.pdf\"")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    test_process_document(pdf_path)
