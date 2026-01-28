"""Configuration for document processing and image analysis via Bedrock."""

# AWS Bedrock configuration for image analysis
IMAGE_ANALYSIS_CONFIG = {
    # Amazon Nova Lite - supports vision/image analysis
    "model_id": "eu.amazon.nova-lite-v1:0",
    "region": "eu-central-1",
    # Inference parameters
    "temperature": 0.3,
    "max_tokens": 2000,
    "top_p": 0.9,
    # Retry configuration
    "max_retries": 3,
    "retry_delay_seconds": 5,
}

# Supported image formats for Bedrock vision models
SUPPORTED_IMAGE_FORMATS = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/webp": "webp",
}

# Maximum image size in bytes (20MB limit for Bedrock)
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024
