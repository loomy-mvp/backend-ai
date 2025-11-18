"""
Simple runner script for Cloud Run Jobs
Runs the batch embedding process and exits
"""

import argparse
import logging
import sys

# Load environment variables from .env file (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    # dotenv not installed - will use system environment variables
    pass

from src.batch_embedder import BatchEmbedder

# Logger will be configured in main after parsing arguments
logger = logging.getLogger(__name__)


def main(bucket_name: str, folders: str, library: str, organization_id: str | None, user_id: str | None, overwrite: bool, log_level: str):
    """Run the batch embedder with configuration"""
    
    # Parse folders from comma-separated string to list
    folder_list = [f.strip() for f in folders.split(',')]
    
    logger.info(f"🚀 Starting batch embedding process")
    logger.info(f"📦 Source bucket: {bucket_name}")
    logger.info(f"📁 Folders: {', '.join(folder_list)}")
    logger.info(f"🏢 Organization: {organization_id or 'N/A'}")
    logger.info(f"📚 Library: {library}")
    if user_id:
        logger.info(f"👤 User: {user_id}")
    logger.info(f"🔄 Overwrite: {overwrite}")
    logger.info(f"📊 Log level: {log_level}")
    
    try:
        # Initialize batch embedder
        embedder = BatchEmbedder(
            library=library,
            organization_id=organization_id,
            user_id=user_id
        )
        
        # Process all files
        stats = embedder.process_files(
            bucket_name=bucket_name,
            folders=folder_list,
            overwrite=overwrite
        )
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("✅ BATCH EMBEDDING COMPLETED")
        logger.info("="*80)
        logger.info(f"📄 Total files:      {stats['total_files']}")
        logger.info(f"✓  Processed:        {stats['processed']}")
        logger.info(f"✗  Failed:           {stats['failed']}")
        logger.info(f"⊘  Skipped:          {stats['skipped']}")
        logger.info(f"📦 Total chunks:     {stats['total_chunks']}")
        logger.info(f"🔢 Total vectors:    {stats['total_vectors']}")
        
        if stats['errors']:
            logger.info(f"\n❌ Errors ({len(stats['errors'])}):")
            for error in stats['errors'][:10]:  # Show first 10 errors
                logger.error(f"  {error['file']} ({error['step']}): {error['error']}")
            if len(stats['errors']) > 10:
                logger.error(f"  ... and {len(stats['errors']) - 10} more errors")
        
        logger.info("="*80)
        
        # Exit with appropriate code
        if stats['failed'] > 0:
            logger.warning(f"⚠️  Completed with {stats['failed']} failures")
            sys.exit(1)
        
        logger.info("✅ All files processed successfully!")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"❌ Batch embedding failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch embed documents from GCS bucket folders into Pinecone"
    )
    parser.add_argument(
        "--bucket-name",
        type=str,
        required=True,
        help="GCS bucket name (e.g., bucket-org123)"
    )
    parser.add_argument(
        "--folders",
        type=str,
        required=True,
        help="Comma-separated list of folder paths within the bucket to process (e.g., 'folder1,folder2,folder3')"
    )
    parser.add_argument(
        "--organization-id",
        type=str,
        required=False,
        default=None,
        help="Organization identifier (required for organization/private libraries)"
    )
    parser.add_argument(
        "--library",
        type=str,
        required=True,
        choices=['organization', 'private', 'public'],
        help="Library type: organization, private, or public"
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default=None,
        help="User identifier (required if library is private)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing vectors in Pinecone"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Configure logging with the specified level
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True  # Override any existing configuration
    )
    
    # Validate combinations
    if args.library in ("organization", "private") and not args.organization_id:
        logger.error("❌ --organization-id is required when --library is organization or private")
        sys.exit(1)
    if args.library == "private" and not args.user_id:
        logger.error("❌ --user-id is required when --library is private")
        sys.exit(1)
    
    main(
        bucket_name=args.bucket_name,
        folders=args.folders,
        library=args.library,
        organization_id=args.organization_id,
        user_id=args.user_id,
        overwrite=args.overwrite,
        log_level=args.log_level
    )
