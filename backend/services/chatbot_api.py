from fastapi import APIRouter, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import List, Any, Optional, Literal
import base64
import httpx
from datetime import datetime
import mimetypes
import os
from pathlib import Path
import re
# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(override=True)

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Retriever
from backend.services.retrieve import Retriever 
retriever = Retriever()

# Configuration
KB_API_BASE_URL = os.getenv("KB_API_BASE_URL", "http://localhost:8000/kb")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
chatbot_webhook_url = os.getenv("CHATBOT_WEBHOOK_URL")
from backend.config.chatbot_config import CHATBOT_CONFIG, SIMILARITY_THRESHOLD
from backend.config.prompts import (
    NO_RAG_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
    CHAT_PROMPT_TEMPLATE,
    TONE_DESCRIPTIONS,
    format_attachment_block,
    format_docs,
    format_context,
    build_question_messages,
)

from backend.utils.ai_workflow_utils.get_config_value import get_config_value
from backend.utils.ai_workflow_utils.get_llm import get_llm
from backend.utils.auth import verify_token
from backend.utils.ai_workflow_utils.create_chain import create_chain
from backend.utils.ai_workflow_utils.get_chat_history import get_chat_history
from backend.utils.ai_workflow_utils.attachment_processing import (
    extract_attachment_text,
    AttachmentProcessingError,
)
from backend.utils.email_notification import send_error_email
from langchain_core.messages import HumanMessage

# Retrieval judge
from backend.utils.ai_workflow_utils.retrieval_judge import RetrievalJudge
retrieval_judge = RetrievalJudge()

chatbot_router = APIRouter(dependencies=[Depends(verify_token)])

# Pydantic models
class ChatMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    messageId: str
    conversationId: str
    userId: str
    organizationId: str
    libraries: Optional[List[str]] = ["organization", "private", "public"]
    sources: Optional[List[str]] = None
    message: str
    attachments: List[str] = None
    test: bool
    tone_of_voice: Literal["formal", "friendly", "technical"] = None

class ChatResponse(BaseModel):
    message_id: str
    status: str
    content: str
    metadata: dict

class RetrieveRequest(BaseModel):
    query: str
    index_name: str
    namespace: str | None = None  # Required when "private" is selected
    libraries: list[str]  # e.g. ["organization", "private", "public"]
    top_k: int = 5
    similarity_threshold: float = SIMILARITY_THRESHOLD
    sources: list[str] | None = None


TEXT_ATTACHMENT_TYPES = {
    "application/pdf",
    "text/plain",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/xml",
    "text/xml",
}
TEXT_ATTACHMENT_EXTENSIONS = {".pdf", ".txt", ".docx", ".xlsx", ".pptx", ".xml"}
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


def _encode_image_data_url(content_type: Optional[str], data: bytes, filename: str) -> str:
    # TODO: To reference a GCS location after persisting the image, modify this function accordingly.
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
        finally:
            # Free the base64 string as soon as it has been decoded
            attachment.pop("data", None)

        resolved_type = (content_type or inferred_type or mimetypes.guess_type(filename)[0] or "").lower()

        if _is_text_attachment(resolved_type, filename):
            try:
                text_payload = extract_attachment_text(file_bytes, filename, resolved_type)
            except AttachmentProcessingError as exc:
                logger.warning("[attachments] Skipping text attachment %s: %s", filename, exc)
                continue
            finally:
                # Free raw bytes immediately after text extraction
                del file_bytes

            normalized_text = text_payload.strip()
            if not normalized_text:
                logger.info("[attachments] No text extracted from %s", filename)
                continue

            text_sections.append(format_attachment_block(filename, normalized_text))
        elif _is_image_attachment(resolved_type, filename):
            try:
                data_url = _encode_image_data_url(resolved_type, file_bytes, filename)
            except Exception as exc:
                logger.warning("[attachments] Failed to encode image %s: %s", filename, exc)
                continue
            finally:
                del file_bytes
            image_inputs.append({
                "filename": filename,
                "data_url": data_url,
            })
        else:
            logger.warning("[attachments] Unsupported attachment type for %s (%s)", filename, resolved_type or "unknown")
            del file_bytes

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
        # Preserve the full data URL so downstream decoding works unchanged.
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


def _normalize_request_attachments(raw_attachments: Optional[List[Any]]) -> List[dict]:
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

async def send_chatbot_webhook(chatbot_webhook_payload: dict):
    """Send chatbot processing status to the configured webhook."""

    if not chatbot_webhook_url:
        print("[webhook] CHATBOT_WEBHOOK_URL not configured; skipping notification")
        return

    try:
        webhook_token = os.getenv("WEBHOOK_TOKEN")
        headers = {"Authorization": f"Bearer {webhook_token}"} if webhook_token else None

        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                chatbot_webhook_url,
                json=chatbot_webhook_payload,
                headers=headers,
            )
            response.raise_for_status()
        print(f"[webhook] Notification sent: status={chatbot_webhook_payload.get('status')} path={chatbot_webhook_payload.get('storage_path')}")
    except Exception as exc:
        print(f"[webhook] Failed to send notification: {exc}")

# Retrieval function from kb_api.py
def retrieve_relevant_docs(
    query: str,
    index_name: str,
    namespace: str,
    libraries: list,
    top_k: int = 5,
    similarity_threshold: float | None = None,
    sources: Optional[List[str]] = None
) -> List[dict]:
    """Retrieve relevant documents from the vector store."""
    print("[retrieve_relevant_docs] - Starts retrieving function")
    try:
        retrieve_request = {
            "query": query,
            "index_name": index_name,
            "namespace": namespace,
            "libraries": libraries,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "sources": sources,
        }
        retrieval = retriever.retrieve(RetrieveRequest(**retrieve_request))
    except Exception as e:
        print(f"[retrieve_relevant_docs] - Exception type: {type(e)}")
        print(f"[retrieve_relevant_docs] - Exception repr: {repr(e)}")
        print(f"[retrieve_relevant_docs] - Exception str: {str(e)}")
        raise

    print(retrieval.get("results", []))
    return retrieval.get("results", [])

async def process_chat_request(chat_data: dict):
    """Background task to process chat request with RAG functionality."""
    try:
        logger.info(f"[process_chat_request] Starting chat processing for message_id {chat_data['message_id']}")
        
        # Extract parameters
        message_id = chat_data["message_id"]
        conversation_id = chat_data["conversation_id"]
        message = chat_data["message"]
        user_id = chat_data["user_id"]
        organization_id = chat_data["organization_id"]
        libraries = chat_data["libraries"]
        source_filter = chat_data.get("sources")
        top_k = chat_data["top_k"]
        index_name = chat_data["index_name"]
        namespace = chat_data["namespace"]
        similarity_threshold = chat_data.get("similarity_threshold")  # Will use default from config if not provided
        attachments = chat_data.get("attachments")
        tone_of_voice = chat_data.get("tone_of_voice", None)
        
        # LLM parameters
        temperature = chat_data["temperature"]
        model = chat_data["model"]
        provider = chat_data["provider"]
        max_tokens = chat_data["max_tokens"]
        
        attachment_context, attachment_count, image_inputs = _build_attachment_context(attachments)
        logger.info(f"[process_chat_request] Processed {attachment_count} text attachments and {len(image_inputs)} images")

        # Initialize LLM
        llm = get_llm(provider, model, temperature, max_tokens)
        logger.info("[process_chat_request] LLM initialized")
        
        # Get chat history
        chat_history = get_chat_history(conversation_id)
        logger.info(f"[process_chat_request] Chat history length: {len(chat_history)}")
        
        # Decide whether to retrieve documents or go plain LLM
        if not chat_history:
            retrieve = True
            logger.info("[process_chat_request] Empty chat history; forcing retrieval")
        else:
            try:
                retrieve = retrieval_judge.judge_retrieval(chat_history, message)
                logger.info(f"[process_chat_request] Retrieval decision: {retrieve}")
            except Exception as e:
                logger.error(f"[process_chat_request] Error during retrieval judgment: {e}")
                retrieve = True  # Default to retrieval on error
        
        # Get tone description for the system prompt
        tone_description = TONE_DESCRIPTIONS.get(tone_of_voice, "")
        
        # If no retrieval, set system message to not use context
        if retrieve is False:
            docs = []
            system_message = NO_RAG_SYSTEM_PROMPT.format(tone_of_voice=tone_description)
            logger.info("[process_chat_request] Retrieval disabled; using NO_RAG_SYSTEM_PROMPT")
        else:
            # Retrieve relevant documents
            system_message = RAG_SYSTEM_PROMPT.format(tone_of_voice=tone_description)
            try:
                docs = retrieve_relevant_docs(
                    query=message,
                    index_name=index_name,
                    namespace=namespace,
                    libraries=libraries,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                    sources=source_filter  # Optional metadata filter
                )
                logger.info(f"[process_chat_request] Retrieved {len(docs)} relevant docs")
            except Exception as e:
                logger.error(f"[process_chat_request] Error during document retrieval: {e}")
                docs = []
        
        # Format documents as context
        formatted_docs = format_docs(docs)
        context = format_context(attachment_context, formatted_docs)
        logger.info("[process_chat_request] Context formatted from attachments and retrieval")
        logger.info("[process_chat_request] Attachment context: " + attachment_context[:50])
        
        # Create RAG chain
        chain = create_chain(llm=llm, prompt_template=CHAT_PROMPT_TEMPLATE)
        logger.info(f"[process_chat_request] {'RAG' if retrieve else 'Plain'} chain created")
        
        question_messages = build_question_messages(message, image_inputs)

        # Get current date
        current_date = datetime.now().strftime("%Y-%m-%d")

        logger.info("[process_chat_request] LLM payload: " + str({
            "system_prompt": system_message,
            "current_date": current_date,
            "context": context,
            "question_messages": "with_images" if image_inputs else "text_only",
            "chat_history": chat_history
        }))
        
        # Generate response
        response = await chain.ainvoke({
            "system_prompt": system_message,
            "current_date": current_date,
            "context": context,
            "question_messages": question_messages,
            "chat_history": chat_history
        })
        response = re.sub(r"cite.*", " ", response)
        logger.info("[process_chat_request] Response generated")
        
        # Prepare sources for response
        retrieved_sources: list[dict] = []
        for doc in docs:
            source_info = {
                "chunk_id": doc.get("chunk_id", None),
                "chunk_text": doc.get("chunk_text", ""),
                "score": doc.get("score", None),
                "page": doc.get("page", None),
                "library": doc.get("library", None),
                "doc_name": doc.get("doc_name", None),
                "storage_path": doc.get("storage_path", None),
                "source": doc.get("source", None)
            }
            retrieved_sources.append(source_info)
        logger.info(f"[process_chat_request] Prepared {len(retrieved_sources)} sources")
        
        # Send webhook notification with success
        await send_chatbot_webhook({
            "message_id": message_id,
            "status": "generated",
            "content": response,
            "metadata": {"chunks": retrieved_sources}
        })
        
        logger.info(f"[process_chat_request] Chat processing completed for message_id {message_id}")
        
    except Exception as e:
        logger.error(f"[process_chat_request] Error processing chat request for message_id {chat_data.get('message_id')}: {str(e)}", exc_info=True)
        
        # Send email notification about the error
        send_error_email(
            subject=f"Chat processing error - {chat_data.get('message_id')}",
            error_details=str(e),
            context={
                "message_id": chat_data.get("message_id"),
                "conversation_id": chat_data.get("conversation_id"),
                "user_id": chat_data.get("user_id"),
                "organization_id": chat_data.get("organization_id"),
            }
        )
        
        # Send webhook notification with error
        await send_chatbot_webhook({
            "message_id": chat_data.get("message_id"),
            "status": "error",
            "content": "",
            "metadata": {
                "error": str(e),
                "conversation_id": chat_data.get("conversation_id")
            }
        })

# API Endpoints
@chatbot_router.post("/chat", response_model=ChatResponse)
async def chat(
    background_tasks: BackgroundTasks,
    request: ChatRequest
):
    """
    Main chat endpoint with RAG functionality.
    
    This endpoint processes chat requests asynchronously:
    1. Validates the request and prepares parameters
    2. Starts background processing
    3. Returns immediately with receipt
    4. Sends webhook notification when processing completes
    
    Args:
        request: ChatRequest containing message, conversation details, and configuration
    
    Returns:
        ChatResponse with status "pending" and empty content
    """
    logger.info(f"[chat] Received chat request for message_id {request.messageId}")
    
    # Message parameters
    message_id = request.messageId
    conversation_id = request.conversationId
    message = request.message
    attachments = request.attachments or []

    # RAG parameters
    user_id = request.userId
    organization_id = request.organizationId
    top_k = get_config_value(config_set=CHATBOT_CONFIG, key="top_k")
    similarity_threshold = SIMILARITY_THRESHOLD
    index_name = str(organization_id)
    namespace = str(user_id)
    libraries = request.libraries
    sources = request.sources

    # LLM parameters
    temperature = get_config_value(config_set=CHATBOT_CONFIG, key="temperature")
    model = get_config_value(config_set=CHATBOT_CONFIG, key="model")
    provider = get_config_value(config_set=CHATBOT_CONFIG, key="provider")
    max_tokens = get_config_value(config_set=CHATBOT_CONFIG, key="max_tokens")
    
    # Handle test mode
    if request.test:
        await send_chatbot_webhook({
            "message_id": message_id,
            "status": "generated",
            "content": "Per calcolare l'IVA al 22%, moltiplica l'imponibile per 0.22. Ad esempio, su 100€ l'IVA è 22€, per un totale di 122€.",
            "metadata": {
                "chunks": [{
                                "chunk_id": "doc1_chunk3",
                                "chunk_text": "Loomy is a digital assistant that helps with various tasks...",
                                "score": 0.95,
                                "page": 5,
                                "library": "public",
                                "doc_name": "loomy_overview.pdf",
                                "storage_path": "fake/path/to/doc1_chunk3"
                            },
                            {
                                "chunk_id": "doc2_chunk7",
                                "chunk_text": "Users can access Loomy from their dashboard after login...",
                                "score": 0.83,
                                "page": 12,
                                "library": "private",
                                "doc_name": "internal_guide.pdf",
                                "storage_path": "fake/path/to/doc2_chunk7"
                            }
                        ]
                }
            })
        return ChatResponse(
            message_id=message_id,
            status="pending",
            content="",
            metadata={}
        )

    # Prompt template workflow - route to document writing
    attachment_payload = _normalize_request_attachments(attachments)
    logger.info(
        "[chat] Normalized %s attachment(s) for processing",
        len(attachment_payload),
    )

    # Prepare chat data for background processing
    chat_data = {
        "message_id": message_id,
        "conversation_id": conversation_id,
        "message": message,
        "user_id": user_id,
        "organization_id": organization_id,
        "similarity_threshold": similarity_threshold,
        "libraries": libraries,
        "sources": sources,
        "top_k": top_k,
        "index_name": index_name,
        "namespace": namespace,
        "temperature": temperature,
        "model": model,
        "provider": provider,
        "max_tokens": max_tokens,
        "attachments": attachment_payload,
        "tone_of_voice": request.tone_of_voice
    }
    
    # Add chat processing to background tasks
    background_tasks.add_task(process_chat_request, chat_data)
    
    # Return immediately with receipt
    return ChatResponse(
        message_id=message_id,
        status="pending",
        content="",
        metadata={
            "conversation_id": conversation_id,
            "message": "Chat request received and processing started. You will receive a webhook notification when complete."
        }
    )

@chatbot_router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now()
    }