"""Batch Embedder for processing multiple documents from GCS to Pinecone"""

import json
import logging
import os
import sys
import mimetypes
from typing import List, Dict, Any
from pathlib import Path

from google.cloud import storage
from google.oauth2 import service_account
import cohere
from pinecone import Pinecone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from backend.config.chatbot_config import EMBEDDING_CONFIG
from backend.utils.ai_workflow_utils.get_config_value import get_config_value
from .kb_helpers import _embed_doc, _upsert_to_vector_store, EmbedRequest, UpsertRequest

logger = logging.getLogger(__name__)


class BatchEmbedder:
    """Handles batch embedding of documents from GCS to Pinecone."""
    
    def __init__(self, library: str, organization_id: str | None = None, user_id: str | None = None):
        if library not in ["organization", "private", "public"]:
            raise ValueError("library must be 'organization', 'private', or 'public'")

        if library in ["organization", "private"] and not organization_id:
            raise ValueError("organization_id is required for organization and private libraries")

        if library == "private" and not user_id:
            raise ValueError("user_id is required for private library")

        self.organization_id = organization_id
        self.library = library
        self.user_id = user_id
        
        # Initialize GCS client with credentials
        # Try to use explicit credentials from env var first, otherwise use default credentials
        gcp_credentials_info = os.getenv("GCP_SERVICE_ACCOUNT_CREDENTIALS")
        if gcp_credentials_info:
            logger.info("Using GCP credentials from environment variable")
            gcp_credentials_info = json.loads(gcp_credentials_info)
            gcp_credentials = service_account.Credentials.from_service_account_info(gcp_credentials_info)
            self.storage_client = storage.Client(credentials=gcp_credentials)
        else:
            logger.info("Using default GCP credentials (Cloud Run service account)")
            self.storage_client = storage.Client()
        
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY environment variable not set")
        self.co = cohere.ClientV2(cohere_api_key)
        self.embedding_model_name = get_config_value(config_set=EMBEDDING_CONFIG, key="model")
        
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        logger.info(
            "BatchEmbedder initialized for org=%s, library=%s, user=%s",
            organization_id or "N/A",
            library,
            user_id or "N/A"
        )
    
    def list_files_in_folders(self, bucket_name: str, folders: List[str]) -> List[Dict[str, str]]:
        """
        List all files in the specified folders (recursively includes subfolders).
        Note: File type filtering is handled by _embed_doc, so all files are listed here.
        """
        logger.info(f"Scanning bucket={bucket_name} for files in folders: {folders}")
        
        try:
            bucket = self.storage_client.get_bucket(bucket_name)
        except Exception as e:
            logger.error(f"Failed to access bucket {bucket_name}: {e}")
            raise
        
        all_files = []
        
        for folder in folders:
            # Ensure folder path format (without trailing slash for prefix)
            folder_path = folder.rstrip('/')
            logger.info(f"Scanning folder: {folder_path}/ (recursively, including all subfolders)")
            
            # List all blobs with this prefix (automatically includes subfolders)
            blobs = bucket.list_blobs(prefix=folder_path)
            
            for blob in blobs:
                # Skip directory markers (blobs ending with /)
                if blob.name.endswith('/'):
                    continue
                
                # Get content type
                content_type = blob.content_type or mimetypes.guess_type(blob.name)[0]
                
                all_files.append({
                    'storage_path': blob.name,
                    'content_type': content_type,
                    'size': blob.size,
                    'updated': blob.updated.isoformat() if blob.updated else None
                })
                logger.debug(f"Found file: {blob.name} (type={content_type}, size={blob.size})")
        
        logger.info(f"Found {len(all_files)} files total (unsupported types will be skipped during processing)")
        return all_files
    
    def process_files(self, bucket_name: str, folders: List[str], overwrite: bool = False) -> Dict[str, Any]:
        files = self.list_files_in_folders(bucket_name, folders)
        
        stats = {
            'total_files': len(files),
            'processed': 0,
            'failed': 0,
            'skipped': 0,
            'total_chunks': 0,
            'total_vectors': 0,
            'errors': []
        }
        
        for i, file_info in enumerate(files, 1):
            storage_path = file_info['storage_path']
            logger.info(f"[{i}/{len(files)}] Processing: {storage_path}")
            
            try:
                embed_request = EmbedRequest(
                    library=self.library,
                    organization_id=self.organization_id,
                    bucket_name=bucket_name,
                    user_id=self.user_id,
                    storage_path=storage_path,
                    content_type=file_info['content_type'],
                    overwrite=overwrite
                )
                
                logger.info(f"  Embedding document...")
                embed_result = _embed_doc(embed_request)
                
                if embed_result.get('status') == 'skipped':
                    logger.info(f"  Skipped (already embedded)")
                    stats['skipped'] += 1
                    continue
                
                if embed_result.get('status') != 'success':
                    error_msg = embed_result.get('message', 'Unknown error')
                    
                    # Check if it's an unsupported file type error
                    if 'No document processor available' in error_msg:
                        logger.warning(f"  Skipped (unsupported file type): {error_msg}")
                        stats['skipped'] += 1
                    else:
                        logger.error(f"  Embedding failed: {error_msg}")
                        stats['failed'] += 1
                        stats['errors'].append({'file': storage_path, 'step': 'embedding', 'error': error_msg})
                    continue
                
                chunks = embed_result.get('chunks', 0)
                vectors = embed_result.get('vectors', [])
                logger.info(f"  Generated {chunks} chunks and {len(vectors)} vectors")
                
                if not vectors:
                    logger.warning(f"  No vectors generated, skipping upsert")
                    stats['skipped'] += 1
                    continue
                
                upsert_request = UpsertRequest(
                    index_name=embed_result['index_name'],
                    namespace=embed_result['namespace'],
                    vectors=vectors
                )
                
                logger.info(f"  Upserting {len(vectors)} vectors to Pinecone...")
                upsert_result = _upsert_to_vector_store(upsert_request)
                
                if upsert_result.get('status') == 'success':
                    upserted = upsert_result.get('upserted', 0)
                    logger.info(f"  ✓ Successfully upserted {upserted} vectors")
                    stats['processed'] += 1
                    stats['total_chunks'] += chunks
                    stats['total_vectors'] += upserted
                else:
                    logger.error(f"  Upsert failed")
                    stats['failed'] += 1
                    stats['errors'].append({'file': storage_path, 'step': 'upsert', 'error': 'Upsert failed'})
                
            except Exception as e:
                logger.error(f"  Error processing file: {e}", exc_info=True)
                stats['failed'] += 1
                stats['errors'].append({'file': storage_path, 'step': 'general', 'error': str(e)})
        
        return stats
