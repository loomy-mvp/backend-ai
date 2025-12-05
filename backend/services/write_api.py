from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import base64
import httpx
import mimetypes
import os
from pathlib import Path

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
from backend.utils.ai_workflow_utils.attachment_processing import (
    extract_attachment_text,
    AttachmentProcessingError,
)

write_router = APIRouter(dependencies=[Depends(verify_token)])

# Environment variables
chatbot_webhook_url = os.getenv("CHATBOT_WEBHOOK_URL")


# Pydantic models
class WriteRequest(BaseModel):
    messageId: str
    conversationId: str
    promptTemplate: str
    message: str
    attachments: List[str] = None
    test: bool = False


class WriteResponse(BaseModel):
    message_id: str
    status: str
    content: str
    metadata: dict


# Attachment processing constants and helpers
TEXT_ATTACHMENT_TYPES = {"application/pdf", "text/plain"}
TEXT_ATTACHMENT_EXTENSIONS = {".pdf", ".txt"}
IMAGE_ATTACHMENT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _decode_attachment_payload(data: str) -> tuple[bytes, Optional[str]]:
    """Decode a base64 payload or data URL and return bytes plus inferred MIME type."""
    if not data:
        raise ValueError("Attachment payload is empty")

    payload = data.strip()
    if payload.startswith("data:"):
        try:
            header, encoded = payload.split(",", 1)
        except ValueError as exc:
            raise ValueError("Malformed data URL for attachment") from exc
        mime = header.split(";")[0].replace("data:", "", 1)
        return base64.b64decode(encoded), mime or None
    return base64.b64decode(payload), None


def _is_text_attachment(content_type: Optional[str], filename: str) -> bool:
    if content_type and content_type.lower() in TEXT_ATTACHMENT_TYPES:
        return True
    extension = Path(filename).suffix.lower()
    return extension in TEXT_ATTACHMENT_EXTENSIONS


def _is_image_attachment(content_type: Optional[str], filename: str) -> bool:
    if content_type and content_type.lower().startswith("image/"):
        return True
    extension = Path(filename).suffix.lower()
    return extension in IMAGE_ATTACHMENT_EXTENSIONS


def _format_attachment_block(filename: str, text: str) -> str:
    return f"""<Attachment>
|Source|: {filename}
|Content|: {text.strip()}"""


def _encode_image_data_url(content_type: Optional[str], data: bytes, filename: str) -> str:
    guessed_type = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{guessed_type};base64,{b64}"


def _build_attachment_context(attachments: Optional[List[dict]]) -> tuple[str, int, List[dict]]:
    """Create textual context and collect image payloads for attachments."""
    if not attachments:
        return "", 0, []

    text_sections: list[str] = []
    image_inputs: list[dict] = []

    for attachment in attachments:
        filename = attachment.get("filename") or "attachment"
        content_type = attachment.get("content_type")
        payload = attachment.get("data")
        if not payload:
            logger.warning("[attachments] Skipping %s: empty payload", filename)
            continue

        try:
            file_bytes, inferred_type = _decode_attachment_payload(payload)
        except Exception as exc:
            logger.warning("[attachments] Failed to decode %s: %s", filename, exc)
            continue

        resolved_type = (content_type or inferred_type or mimetypes.guess_type(filename)[0] or "").lower()

        if _is_text_attachment(resolved_type, filename):
            try:
                text_payload = extract_attachment_text(file_bytes, filename, resolved_type)
            except AttachmentProcessingError as exc:
                logger.warning("[attachments] Skipping text attachment %s: %s", filename, exc)
                continue

            normalized_text = text_payload.strip()
            if not normalized_text:
                logger.info("[attachments] No text extracted from %s", filename)
                continue

            text_sections.append(_format_attachment_block(filename, normalized_text))
        elif _is_image_attachment(resolved_type, filename):
            data_url = _encode_image_data_url(resolved_type, file_bytes, filename)
            image_inputs.append({
                "filename": filename,
                "data_url": data_url,
            })
        else:
            logger.warning("[attachments] Unsupported attachment type for %s (%s)", filename, resolved_type or "unknown")

    text_context = "\n----------\n".join(text_sections)
    return text_context, len(text_sections), image_inputs


def _guess_attachment_filename(content_type: Optional[str], index: int) -> str:
    """Derive a fallback filename when the client does not provide one."""
    extension = mimetypes.guess_extension(content_type or "") or ""
    return f"attachment_{index + 1}{extension}"


def _string_attachment_to_dict(raw_attachment: str, index: int) -> dict:
    """Convert a simple string payload into the structured attachment shape."""
    payload = raw_attachment.strip()
    if not payload:
        raise ValueError("Attachment string is empty")

    content_type: Optional[str] = None
    data = payload

    if payload.startswith("data:"):
        header = payload.split(";", 1)[0]
        content_type = header.replace("data:", "", 1) or None
    elif ":" in payload:
        possible_type, remainder = payload.split(":", 1)
        if "/" in possible_type:
            content_type = possible_type
            data = remainder

    return {
        "filename": _guess_attachment_filename(content_type, index),
        "content_type": content_type,
        "data": data,
    }


def _normalize_request_attachments(raw_attachments: Optional[List[str]]) -> List[dict]:
    """Ensure every attachment passed downstream is a dict with expected keys."""
    if not raw_attachments:
        return []

    normalized: List[dict] = []

    for idx, attachment in enumerate(raw_attachments):
        if not isinstance(attachment, str):
            logger.warning(
                "[attachments] Expected string payload but received %s at index %s",
                type(attachment),
                idx,
            )
            continue

        try:
            payload = _string_attachment_to_dict(attachment, idx)
            normalized.append(payload)
        except Exception as exc:
            logger.warning(
                "[attachments] Failed to normalize attachment at index %s: %s",
                idx,
                exc,
            )

    return normalized


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
        attachments = write_data.get("attachments")
        
        # LLM parameters
        llm_params = {
            "provider": write_data["provider"],
            "model": write_data["model"],
            "temperature": write_data["temperature"],
            "max_tokens": write_data["max_tokens"]
        }
        
        # Process attachments
        attachment_context, attachment_count, image_inputs = _build_attachment_context(attachments)
        logger.info(f"[process_write_request] Processed {attachment_count} text attachments and {len(image_inputs)} images")
        
        # If there are text attachments, append them to the message
        if attachment_context:
            message = f"{message}\n\n<<<User Attachments>>>\n{attachment_context}"
        
        # Generate document using Writer
        response = await writer.write_document(
            message=message,
            template=template,
            conversation_id=conversation_id,
            llm_params=llm_params,
            image_inputs=image_inputs
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
        request: WriteRequest containing message, template, and configuration
    
    Returns:
        WriteResponse with status "pending" and empty content
    """
    logger.info(f"[write] Received write request for message_id {request.messageId}")
    
    # Extract parameters
    message_id = request.messageId
    conversation_id = request.conversationId
    message = request.message
    template = request.promptTemplate
    attachments = request.attachments or []
    
    # LLM parameters from config
    temperature = get_config_value(config_set=CHATBOT_CONFIG, key="temperature")
    model = get_config_value(config_set=CHATBOT_CONFIG, key="model")
    provider = get_config_value(config_set=CHATBOT_CONFIG, key="provider")
    max_tokens = get_config_value(config_set=CHATBOT_CONFIG, key="max_tokens")
    
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
    
    # Normalize attachments
    attachment_payload = _normalize_request_attachments(attachments)
    logger.info(
        "[write] Normalized %s attachment(s) for writing",
        len(attachment_payload),
    )
    
    # Prepare write data for background processing
    write_data = {
        "message_id": message_id,
        "conversation_id": conversation_id,
        "message": message,
        "template": template,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "attachments": attachment_payload
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
