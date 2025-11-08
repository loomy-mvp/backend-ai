from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import httpx
from backend.config.prompts import NO_RAG_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT
from langchain.prompts import ChatPromptTemplate #, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from datetime import datetime
import os
from enum import Enum
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
from backend.utils.get_config_value import get_config_value
from backend.utils.auth import verify_token

# DB connection
from backend.utils.db_utils import DBUtils

chatbot_router = APIRouter(dependencies=[Depends(verify_token)])

# Enums for model providers
class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"

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
    retrieve: Optional[bool] = True # ! Default to be left out
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

# LLM initialization functions
def get_llm(provider: ModelProvider = None, model: str = None, temperature: float = 0, max_tokens: int = None):
    """Initialize LLM based on provider and model using config, with fallback to defaults."""
    import importlib
    
    # Mapping of providers to their LangChain modules and classes
    provider_mapping = {
        ModelProvider.OPENAI: ("langchain_openai", "ChatOpenAI", "max_tokens"),
        ModelProvider.ANTHROPIC: ("langchain_anthropic", "ChatAnthropic", "max_tokens"),
        ModelProvider.GOOGLE: ("langchain_google_vertexai", "ChatVertexAI", "max_output_tokens"),
        # Add more providers as needed
    }
    
    if provider is None:
        provider = get_config_value(config_set=CHATBOT_CONFIG, key="provider")
    if model is None:
        model = get_config_value(config_set=CHATBOT_CONFIG, key="model")
    if max_tokens is None:
        max_tokens = get_config_value(config_set=CHATBOT_CONFIG, key="max_tokens")

    if provider not in provider_mapping:
        raise ValueError(f"Unsupported provider: {provider}")
    
    module_name, class_name, max_tokens_param = provider_mapping[provider]
    
    try:
        # Dynamically import the module and get the class
        module = importlib.import_module(module_name)
        llm_class = getattr(module, class_name)
        
        # Prepare kwargs with the correct parameter name for max_tokens
        kwargs = {
            "model": model,
            "temperature": temperature,
            max_tokens_param: max_tokens
        }
        
        return llm_class(**kwargs)
        
    except ImportError as e:
        raise ValueError(f"Failed to import {module_name}: {str(e)}. Make sure the package is installed.")
    except Exception as e:
        raise ValueError(f"Failed to initialize LLM for provider {provider} with model {model}: {str(e)}")

def get_chat_history(conversation_id: str) -> List[Dict[str, Any]]:
    """Get conversation memory for a conversation from the database"""
    db_results = DBUtils.execute_query(
        "SELECT * FROM messages WHERE conversation_id = %s",
        (conversation_id,)
    )

    messages = []
    for msg in db_results:
        if msg[6] != 'error':
            message_content = {
                "sender": msg[2],
                "content": msg[3],
                "metadata": msg[4]
                # "created_at": msg[5]
            }
            messages.append(message_content)
    
    return messages

def format_docs(docs: List[dict]) -> str:
    """Format retrieved documents for the prompt."""
    if not docs:
        return "No relevant documents found."
    
    formatted_docs = []
    for i, doc in enumerate(docs, 1):
        content = doc.get("chunk_text", "")
        source = doc.get("doc_name", "")
        
        formatted_doc = f"""<Document {i}>
        |Source|: {source}
        |Content|: {content}"""
        formatted_docs.append(formatted_doc)
    
    return "\n----------\n".join(formatted_docs)

def create_rag_chain(llm, system_message: str = None):
    """Create the RAG chain with LangChain."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message or ""),
        ("user", "<<<Chat history>>>\n{chat_history}\n"), # ? can use MessagesPlaceholder(variable_name="chat_history"),
        ("user", "<<<Context>>>\n{context}\n"),
        ("user", "<<<Prompt>>>\n{question}")
    ])
    
    chain = (
        RunnablePassthrough()
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

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

    # RAG Chatbot workflow
    try:
        print("[chat] Start chat endpoint")
        # If no retrieval, set system message to not use context
        if request.retrieve is False:
            docs = []
            system_message = NO_RAG_SYSTEM_PROMPT
            print("[chat] Retrieval disabled; using NO_RAG_SYSTEM_PROMPT")
        else:
            # Retrieve relevant documents
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
        chain = create_rag_chain(llm, system_message)
        print("[chat] RAG chain created")
        
        # Get chat history
        chat_history = get_chat_history(conversation_id)
        print(f"[chat] Chat history length: {len(chat_history)}")
        
        # Generate response (Memory persistence to DB is handled by TypeSript backend)
        response = await chain.ainvoke({
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