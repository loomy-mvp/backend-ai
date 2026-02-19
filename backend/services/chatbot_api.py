from fastapi import APIRouter, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Literal
import httpx
from datetime import datetime
import os
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
    _build_attachment_context,
    _normalize_request_attachments,
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
        
    except AttachmentProcessingError as e:
        logger.error(f"[process_chat_request] Attachment processing error for message_id {chat_data.get('message_id')}: {str(e)}")
        await send_chatbot_webhook({
            "message_id": chat_data.get("message_id"),
            "status": "error",
            "content": "Error elaborating attachments",
            "metadata": {
                "error": str(e),
                "conversation_id": chat_data.get("conversation_id")
            }
        })

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