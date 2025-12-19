from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime
import httpx
import os

from dotenv import load_dotenv
load_dotenv(override=True)

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Writer service
from backend.services.write import Writer
writer = Writer()

# Configuration
from backend.config.chatbot_config import CHATBOT_CONFIG
from backend.utils.ai_workflow_utils.get_config_value import get_config_value
from backend.utils.auth import verify_token

write_router = APIRouter(dependencies=[Depends(verify_token)])

# Environment variables
chatbot_webhook_url = os.getenv("CHATBOT_WEBHOOK_URL")


# Pydantic models
class WriteRequest(BaseModel):
    messageId: str
    conversationId: str
    promptTemplate: Optional[str] = None
    message: Optional[str] = None
    requirements: Optional[Any] = None  # JSON formatted string with template field values
    test: bool = False


class WriteResponse(BaseModel):
    message_id: str
    status: str
    content: str
    metadata: dict


async def send_write_webhook(webhook_payload: dict):
    """Send document writing status to the configured webhook."""
    if not chatbot_webhook_url:
        print("[webhook] CHATBOT_WEBHOOK_URL not configured; skipping notification")
        return

    try:
        webhook_token = os.getenv("WEBHOOK_TOKEN")
        headers = {"Authorization": f"Bearer {webhook_token}"} if webhook_token else None

        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                chatbot_webhook_url,
                json=webhook_payload,
                headers=headers,
            )
            response.raise_for_status()
        print(f"[webhook] Notification sent: status={webhook_payload.get('status')}")
    except Exception as exc:
        print(f"[webhook] Failed to send notification: {exc}")


async def process_write_request(write_data: dict):
    """Background task to process document writing request with template."""
    try:
        logger.info(f"[process_write_request] Starting document writing for message_id {write_data['message_id']}")
        
        # Extract parameters
        message_id = write_data["message_id"]
        conversation_id = write_data["conversation_id"]
        message = write_data["message"]
        template = write_data["template"]
        requirements = write_data.get("requirements", "")
        
        # LLM parameters
        llm_params = {
            "provider": write_data["provider"],
            "model": write_data["model"],
            "temperature": write_data["temperature"],
            "max_tokens": write_data["max_tokens"]
        }
        
        # Generate document using Writer
        response = await writer.write_document(
            message=message,
            template=template,
            requirements=requirements,
            conversation_id=conversation_id,
            llm_params=llm_params
        )
        logger.info("[process_write_request] Document generated")
        
        # Send webhook notification with success
        await send_write_webhook({
            "message_id": message_id,
            "status": "generated",
            "content": response,
            "metadata": {"template_used": True}
        })
        
        logger.info(f"[process_write_request] Document writing completed for message_id {message_id}")
        
    except Exception as e:
        logger.error(f"[process_write_request] Error processing write request for message_id {write_data.get('message_id')}: {str(e)}", exc_info=True)
        
        # Send webhook notification with error
        await send_write_webhook({
            "message_id": write_data.get("message_id"),
            "status": "error",
            "content": "",
            "metadata": {
                "error": str(e),
                "conversation_id": write_data.get("conversation_id")
            }
        })


# API Endpoints
@write_router.post("/write", response_model=WriteResponse)
async def write_document(
    background_tasks: BackgroundTasks,
    request: WriteRequest
):
    """
    Document writing endpoint with template functionality.
    
    This endpoint processes document writing requests asynchronously:
    1. Validates the request and prepares parameters
    2. Starts background processing
    3. Returns immediately with receipt
    4. Sends webhook notification when processing completes
    
    Args:
        request: WriteRequest containing message, template, requirements and configuration
    
    Returns:
        WriteResponse with status "pending" and empty content
    """
    logger.info(f"[write] Received write request for message_id {request.messageId}")
    
    # Extract parameters
    message_id = request.messageId
    conversation_id = request.conversationId
    message = request.message
    template = request.promptTemplate
    requirements = request.requirements or ""
    
    # LLM parameters from config
    temperature = get_config_value(config_set=CHATBOT_CONFIG, key="temperature")
    model = get_config_value(config_set=CHATBOT_CONFIG, key="model")
    provider = get_config_value(config_set=CHATBOT_CONFIG, key="provider")
    max_tokens = get_config_value(config_set=CHATBOT_CONFIG, key="write_max_tokens")
    
    # Handle test mode
    if request.test:
        await send_write_webhook({
            "message_id": message_id,
            "status": "generated",
            "content": "Test document generated successfully based on the provided template.",
            "metadata": {"template_used": True}
        })
        return WriteResponse(
            message_id=message_id,
            status="pending",
            content="",
            metadata={}
        )
    
    # Prepare write data for background processing
    write_data = {
        "message_id": message_id,
        "conversation_id": conversation_id,
        "message": message,
        "template": template,
        "requirements": requirements,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    # Add write processing to background tasks
    background_tasks.add_task(process_write_request, write_data)
    
    # Return immediately with receipt
    return WriteResponse(
        message_id=message_id,
        status="pending",
        content="",
        metadata={
            "conversation_id": conversation_id,
            "message": "Document writing request received and processing started. You will receive a webhook notification when complete."
        }
    )


@write_router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now()
    }
