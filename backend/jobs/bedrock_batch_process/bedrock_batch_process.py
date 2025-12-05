"""
Batch process txt files from GCS using AWS Bedrock Nova Micro with prompt caching.

Reads all .txt files from loomy-jobs/{folder} in Google Cloud Storage,
processes each through AWS Bedrock using the Converse API with a user-defined
prompt, and saves outputs as {original_filename}_output.txt.

Designed to run on Google Cloud Run with GCS for storage and AWS Bedrock for inference.
"""

import argparse
import json
import logging
import os
import sys

import boto3
from botocore.config import Config
from google.cloud import storage as gcs
from google.oauth2 import service_account
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

logger.info("Started loading bedrock_batch_process.py")

# Default values
DEFAULT_BUCKET_NAME = "loomy-jobs"
# Amazon Nova Micro (EU - Frankfurt region)
DEFAULT_MODEL_ID = "eu.amazon.nova-micro-v1:0"
# Default inference config
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 5000
DEFAULT_TOP_P = 0.9


def get_gcs_client():
    """Create and return a Google Cloud Storage client.
    
    Uses GCP_SERVICE_ACCOUNT_CREDENTIALS env var if available (for local dev),
    otherwise uses default credentials (for Cloud Run).
    """
    gcp_credentials_info = os.getenv("GCP_SERVICE_ACCOUNT_CREDENTIALS")
    if gcp_credentials_info:
        gcp_credentials_info = json.loads(gcp_credentials_info)
        gcp_service_account_credentials = service_account.Credentials.from_service_account_info(
            gcp_credentials_info
        )
        return gcs.Client(credentials=gcp_service_account_credentials)
    else:
        # Use default credentials (e.g., Cloud Run service account)
        return gcs.Client()


def get_bedrock_client():
    """Create and return a Bedrock Runtime client with retry configuration."""
    config = Config(
        region_name=os.getenv("AWS_BEDROCK_REGION", "eu-central-1"),
        retries={"max_attempts": 3, "mode": "adaptive"},
    )
    return boto3.client("bedrock-runtime", config=config)


def list_txt_files(gcs_client, bucket_name: str, folder: str) -> list[str]:
    """List all .txt files in the specified GCS folder."""
    bucket = gcs_client.bucket(bucket_name)
    prefix = f"{folder}/"
    txt_files = []

    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        # Only include .txt files, exclude _output.txt files
        if blob.name.endswith(".txt") and not blob.name.endswith("_output.txt"):
            txt_files.append(blob.name)

    return txt_files


def read_file_from_gcs(gcs_client, bucket_name: str, blob_name: str) -> str:
    """Read and return the content of a file from GCS."""
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_text(encoding="utf-8")


def save_output_to_gcs(gcs_client, bucket_name: str, blob_name: str, content: str) -> None:
    """Save content to GCS as a txt file."""
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(
        content.encode("utf-8"), 
        content_type="text/plain; charset=utf-8"
    )


def process_with_bedrock(
    bedrock_client,
    model_id: str,
    user_prompt: str,
    file_content: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    top_p: float = DEFAULT_TOP_P,
    cache_point_type: str = "default",
) -> tuple[str, dict]:
    """
    Process content using AWS Bedrock Converse API with prompt caching.

    The user_prompt is placed first with a cache checkpoint, so it can be
    cached and reused across multiple calls with different file contents.

    Args:
        bedrock_client: Boto3 Bedrock Runtime client
        model_id: The Bedrock model ID to use
        user_prompt: The user-defined prompt (cached across calls)
        file_content: The content of the txt file to process
        temperature: Controls randomness (0.0-1.0, default 0.7)
        max_tokens: Maximum tokens to generate (default 4096)
        top_p: Nucleus sampling parameter (0.0-1.0, default 0.9)
        cache_point_type: Type of cache point ("default" for standard caching)

    Returns:
        Tuple containing the model's response text and the usage metadata returned by Bedrock.
    """
    # Build the message with the user prompt at the top (for caching)
    # followed by the file content
    # The cachePoint is placed after the user_prompt to enable prompt caching
    messages = [
        {
            "role": "user",
            "content": [
                # User-defined prompt - this part will be cached
                {
                    "text": user_prompt,
                },
                # Cache checkpoint - marks the boundary for caching
                # Everything before this point will be cached
                {
                    "cachePoint": {
                        "type": cache_point_type,
                    },
                },
                # File content - this varies per file and won't be cached
                {
                    "text": f"\n\n{file_content}",
                },
            ],
        }
    ]

    try:
        response = bedrock_client.converse(
            modelId=model_id,
            messages=messages,
            inferenceConfig={
                "temperature": temperature,
                "maxTokens": max_tokens,
                "topP": top_p,
            },
        )

        # Extract the response text
        output_message = response.get("output", {}).get("message", {})
        content_blocks = output_message.get("content", [])

        response_text = ""
        for block in content_blocks:
            if "text" in block:
                response_text += block["text"]

        # Log cache usage information if available
        usage = (response.get("usage", {}) or {}).copy()
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)
        cache_read_tokens = usage.get("cacheReadInputTokens", 0)
        cache_write_tokens = usage.get("cacheWriteInputTokens", 0)
        non_cached_input_tokens = max(input_tokens - cache_read_tokens, 0)
        usage["nonCachedInputTokens"] = non_cached_input_tokens

        logger.info(
            "Token usage (non-cached) - Input: %d, Output: %d",
            non_cached_input_tokens,
            output_tokens,
        )
        if cache_read_tokens or cache_write_tokens:
            logger.info(
                "Cache usage - Read: %d tokens, Write: %d tokens",
                cache_read_tokens,
                cache_write_tokens,
            )
        logger.info("RESPONSE TEXT: %s", response_text)
        return response_text, usage

    except Exception as e:
        logger.error("Bedrock API call failed: %s", str(e))
        raise


def get_output_key(original_key: str, folder: str) -> str:
    """Generate the output file key in the output subfolder.
    
    Example: folder/file.txt -> folder/output/file_output.txt
    """
    # Get the filename from the original key
    filename = original_key.split("/")[-1]
    
    # Remove .txt extension and add _output.txt
    if filename.endswith(".txt"):
        output_filename = filename[:-4] + "_output.txt"
    else:
        output_filename = filename + "_output.txt"
    
    # Place in output subfolder
    return f"{folder}/output/{output_filename}"


def main() -> None:
    """Main entry point for the batch processing job."""
    parser = argparse.ArgumentParser(
        description="Process txt files from GCS using AWS Bedrock Nova Micro"
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Folder name within the loomy-jobs bucket to process",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="User-defined prompt to use for processing each file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files without processing them",
    )
    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET_NAME,
        help=f"GCS bucket name (default: {DEFAULT_BUCKET_NAME})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"Bedrock model ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Temperature for generation (0.0-1.0, default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens to generate (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help=f"Top-p nucleus sampling (0.0-1.0, default: {DEFAULT_TOP_P})",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    logger.info("Starting batch processing job")
    logger.info("Bucket: %s", args.bucket)
    logger.info("Folder: %s", args.folder)
    logger.info("Model: %s", args.model)
    logger.info("Prompt: %s", args.prompt[:100] + "..." if len(args.prompt) > 100 else args.prompt)

    gcs_client = get_gcs_client()
    bedrock_client = get_bedrock_client()

    # List all txt files in the folder
    txt_files = list_txt_files(gcs_client, args.bucket, args.folder)

    if not txt_files:
        logger.warning("No .txt files found in %s/%s", args.bucket, args.folder)
        return

    logger.info("Found %d .txt files to process", len(txt_files))

    if args.dry_run:
        logger.info("Dry run mode - listing files only:")
        for file_path in txt_files:
            logger.info("  - %s", file_path)
        return

    # Process each file
    success_count = 0
    error_count = 0
    total_non_cached_input_tokens = 0
    total_output_tokens = 0
    token_reporting_count = 0

    for i, file_path in enumerate(txt_files, 1):
        logger.info("Processing file %d/%d: %s", i, len(txt_files), file_path)

        try:
            # Read file content
            file_content = read_file_from_gcs(gcs_client, args.bucket, file_path)
            logger.info("Read %d characters from file", len(file_content))

            # Skip processing if file is empty, create empty output
            usage = None
            if not file_content.strip():
                logger.info("File is empty, creating empty output file")
                output = ""
            else:
                # Process with Bedrock
                output, usage = process_with_bedrock(
                    bedrock_client,
                    args.model,
                    args.prompt,
                    file_content,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    top_p=args.top_p,
                )

            # Save output
            output_path = get_output_key(file_path, args.folder)
            save_output_to_gcs(gcs_client, args.bucket, output_path, output)
            logger.info("Saved output to %s", output_path)

            if usage:
                total_non_cached_input_tokens += usage.get("nonCachedInputTokens", 0)
                total_output_tokens += usage.get("outputTokens", 0)
                token_reporting_count += 1

            success_count += 1

        except Exception as e:
            logger.error("Failed to process %s: %s", file_path, str(e))
            error_count += 1
            continue

    logger.info(
        "Batch processing complete. Success: %d, Errors: %d",
        success_count,
        error_count,
    )

    if token_reporting_count:
        mean_input_tokens = total_non_cached_input_tokens / token_reporting_count
        mean_output_tokens = total_output_tokens / token_reporting_count
        logger.info(
            "Average non-cached tokens per processed file - Input: %.2f, Output: %.2f (%d files)",
            mean_input_tokens,
            mean_output_tokens,
            token_reporting_count,
        )

    if error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
