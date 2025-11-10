from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import httpx
from datetime import datetime
import os
# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(override=True)

# Retriever
from backend.services.retrieve import Retriever 
from backend.services.write import Writer
retriever = Retriever()
writer = Writer()

# Configuration
KB_API_BASE_URL = os.getenv("KB_API_BASE_URL", "http://localhost:8000/kb")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
chatbot_webhook_url = os.getenv("CHATBOT_WEBHOOK_URL")
from backend.config.chatbot_config import CHATBOT_CONFIG
from backend.config.prompts import NO_RAG_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT, CHAT_PROMPT_TEMPLATE
from backend.utils.ai_workflow_utils.get_config_value import get_config_value
from backend.utils.ai_workflow_utils.get_llm import get_llm
from backend.utils.auth import verify_token
from backend.utils.ai_workflow_utils.create_chain import create_chain
from backend.utils.ai_workflow_utils.get_chat_history import get_chat_history

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
    message: str
    promptTemplate: Optional[str] = None
    test: bool

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
def retrieve_relevant_docs(query: str, index_name: str, namespace: str, libraries: list, top_k: int = 5) -> List[dict]:
    """Retrieve relevant documents from the vector store."""
    print("[retrieve_relevant_docs] - Starts retrieving function")
    try:
        retrieve_request = {
            "query": query,
            "index_name": index_name,
            "namespace": namespace,
            "libraries": libraries,
            "top_k": top_k
        }
        retrieval = retriever.retrieve(RetrieveRequest(**retrieve_request))
    except Exception as e:
        print(f"[retrieve_relevant_docs] - Exception type: {type(e)}")
        print(f"[retrieve_relevant_docs] - Exception repr: {repr(e)}")
        print(f"[retrieve_relevant_docs] - Exception str: {str(e)}")
        raise

    print(retrieval.get("results", []))
    return retrieval.get("results", [])

def format_docs(docs: List[dict]) -> str:
    """Format retrieved documents for the prompt."""
    if not docs:
        return ""
    
    formatted_docs = []
    for i, doc in enumerate(docs, 1):
        content = doc.get("chunk_text", "")
        source = doc.get("doc_name", "")
        
        formatted_doc = f"""<Document {i}>
        |Source|: {source}
        |Content|: {content}"""
        formatted_docs.append(formatted_doc)
    
    return "\n----------\n".join(formatted_docs)

# API Endpoints
@chatbot_router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint with RAG functionality.
    """
    # Message parameters
    message_id = request.messageId
    conversation_id = request.conversationId
    message = request.message
    print("[chat] QUERY: ", message)
    prompt_template = request.promptTemplate

    # RAG parameters
    user_id = request.userId
    organization_id = request.organizationId
    top_k = get_config_value(config_set=CHATBOT_CONFIG, key="top_k")
    index_name = str(organization_id)
    namespace = str(user_id)
    libraries = request.libraries

    # LLM parameters
    system_message = None
    temperature = get_config_value(config_set=CHATBOT_CONFIG, key="temperature")
    model = get_config_value(config_set=CHATBOT_CONFIG, key="model")
    provider = get_config_value(config_set=CHATBOT_CONFIG, key="provider")
    max_tokens = get_config_value(config_set=CHATBOT_CONFIG, key="max_tokens")
    
    # Initialize LLM
    llm = get_llm(provider, model, temperature, max_tokens)
    print("[chat] LLM initialized")
    
    # Get chat history
    chat_history = get_chat_history(conversation_id)
    print(f"[chat] Chat history length: {len(chat_history)}")

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
            status="generating",
            content="",
            metadata={}
        )

    # Prompt template workflow
    if prompt_template:
        pass  # TODO: Implement custom prompt template handling

    # Decide whether to retrieve documents or go plain LLM
    retrieve = retrieval_judge.judge_retrieval(chat_history, message)

    # RAG Chatbot workflow
    try:
        print("[chat] Start chat endpoint")
        # If no retrieval, set system message to not use context
        if retrieve is False:
            docs = []
            system_message = NO_RAG_SYSTEM_PROMPT
            print("[chat] Retrieval disabled; using NO_RAG_SYSTEM_PROMPT")
        else:
            # Retrieve relevant documents
            # ? Retrieve va fatto anche sulla history?
            system_message = RAG_SYSTEM_PROMPT
            docs = retrieve_relevant_docs(
                query=message,
                index_name=index_name,
                namespace=namespace,
                libraries=libraries,
                top_k=top_k
            )
            print(f"[chat] Retrieved {len(docs)} relevant docs")
        
        # Format documents as context
        context = format_docs(docs)
        print("[chat] Context formatted")
        
        # Create RAG chain
        chain = create_chain(llm=llm, prompt_template=CHAT_PROMPT_TEMPLATE)
        print(f"""[chat] {'RAG' if retrieve else 'Plain'} chain created""")
        
        # Generate response (Memory persistence to DB is handled by TypeSript backend)
        response = await chain.ainvoke({
            "system_prompt": system_message,
            "context": context,
            "question": message,
            "chat_history": chat_history
        })
        print("[chat] Response generated")
        
        # Prepare sources for response
        sources = []
        for doc in docs:
            source_info = {
                "chunk_id": doc.get("chunk_id", None),
                "chunk_text": doc.get("chunk_text", ""),
                "score": doc.get("score", None),
                "page": doc.get("page", None),
                "library": doc.get("library", None),
                "doc_name": doc.get("doc_name", None),
                "storage_path": doc.get("storage_path", None)
            }
            sources.append(source_info)
        print(f"[chat] Returning response with {len(sources)} sources")
        
        # Send webhook notification
        await send_chatbot_webhook({
            "message_id": message_id,
            "status": "generated",
            "content": response,
            "metadata": {
                "chunks": sources
            }
        })
        
        return ChatResponse(
            message_id=message_id,
            status="generated",
            content=response,
            metadata={"chunks": sources}
        )
        
    except Exception as e:
        print(f"[chat] Exception occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@chatbot_router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now()
    }