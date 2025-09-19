from fastapi import FastAPI, APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import httpx
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
# LLM providers are now loaded dynamically in get_llm function
import uuid
from datetime import datetime
import os
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Configuration
KB_API_BASE_URL = os.getenv("KB_API_BASE_URL", "http://localhost:8000/kb")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
from backend.config.chatbot_config import CHATBOT_CONFIG

# Helper to get config value with fallback to default
def get_config_value(key: str):
    if CHATBOT_CONFIG.get(key) is not None:
        return CHATBOT_CONFIG[key]
    return CHATBOT_CONFIG.get(f"default_{key}")

from backend.utils.auth import verify_token

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
    message: str
    session_id: Optional[str] = None
    index_name: str
    top_k: Optional[int] = 5
    system_message: Optional[str] = None
    temperature: Optional[float] = 0
    model: Optional[str] = get_config_value("model")
    provider: Optional[ModelProvider] = get_config_value("provider")
    max_tokens: Optional[int] = get_config_value("max_tokens")

class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: List[Dict[str, Any]]
    timestamp: datetime

class SessionInfo(BaseModel):
    session_id: str
    created_at: datetime
    message_count: int
    last_activity: datetime

class SessionListResponse(BaseModel):
    sessions: List[SessionInfo]

class DeleteSessionResponse(BaseModel):
    message: str
    session_id: str

# In-memory storage for sessions (in production, use Redis or database)
chat_sessions: Dict[str, ConversationBufferWindowMemory] = {}
session_metadata: Dict[str, Dict[str, Any]] = {}

# Retrieval function from kb_api.py
async def retrieve_relevant_docs(query: str, index_name: str, top_k: int = 5) -> List[dict]:
    """Retrieve relevant documents from the vector store."""
    print("[retrieve_relevant_docs] - Starts retrieving function")
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            headers = {}
            api_token = os.getenv("API_TOKEN") or os.getenv("LOOMY_API_TOKEN")
            if api_token:
                headers["Authorization"] = f"Bearer {api_token}"
            response = await client.post(
                f"{KB_API_BASE_URL}/retrieve",
                json={
                    "query": query,
                    "index_name": index_name,
                    "top_k": top_k
                },
                headers=headers
            )
            print("[retrieve_relevant_docs] - Post request made")
        except Exception as e:
            print(f"[retrieve_relevant_docs] - Exception type: {type(e)}")
            print(f"[retrieve_relevant_docs] - Exception repr: {repr(e)}")
            print(f"[retrieve_relevant_docs] - Exception str: {str(e)}")
            raise
        if response.status_code != 200:
            print("[retrieve_relevant_docs] - Failed to retrieve")
            raise HTTPException(status_code=response.status_code, detail="Failed to retrieve documents")
        
        data = response.json()
        print(data.get("results", []))
        return data.get("results", [])

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
        provider = get_config_value("provider")
    if model is None:
        model = get_config_value("model")
    if max_tokens is None:
        max_tokens = get_config_value("max_tokens")

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

def get_memory(session_id: str, k: int = 10) -> ConversationBufferWindowMemory:
    """Get or create conversation memory for a session."""
    if session_id not in chat_sessions:
        chat_sessions[session_id] = ConversationBufferWindowMemory(
            k=k,
            memory_key="chat_history",
            return_messages=True
        )
        session_metadata[session_id] = {
            "created_at": datetime.now(),
            "message_count": 0,
            "last_activity": datetime.now()
        }
    return chat_sessions[session_id]

def format_docs(docs: List[dict]) -> str:
    """Format retrieved documents for the prompt."""
    if not docs:
        return "No relevant documents found."
    
    formatted_docs = []
    for i, doc in enumerate(docs, 1):
        content = doc.get("chunk_text", "")
        metadata = doc.get("metadata", {})
        
        formatted_doc = f"Document {i}:\nContent: {content}\n" # TODO: add the source from metadata after getting the right key -- \nSource: {source}
        formatted_docs.append(formatted_doc)
    
    return "\n---\n".join(formatted_docs)

def create_rag_chain(llm, system_message: str = None):
    """Create the RAG chain with LangChain."""
    
    default_system_message = """You are a helpful AI assistant. Use the following context to answer the user's question. 
If you cannot answer the question based on the context provided, say so clearly.
Always be accurate and cite the sources when possible.

Context:
{context}"""
    
    system_msg = system_message or default_system_message
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
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
    """Main chat endpoint with RAG functionality."""
    try:
        print("[chat] Start chat endpoint")
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        print(f"[chat] Session ID: {session_id}")
        
        # Get or create memory for this session
        memory = get_memory(session_id)
        print("[chat] Memory retrieved")
        
        # Retrieve relevant documents
        docs = await retrieve_relevant_docs(
            query=request.message,
            index_name=request.index_name,
            top_k=request.top_k
        )
        print(f"[chat] Retrieved {len(docs)} relevant docs")
        
        # Format documents as context
        context = format_docs(docs)
        print("[chat] Context formatted")
        
        # Initialize LLM
        llm = get_llm(request.provider, request.model, request.temperature, request.max_tokens)
        print("[chat] LLM initialized")
        
        # Create RAG chain
        chain = create_rag_chain(llm, request.system_message)
        print("[chat] RAG chain created")
        
        # Get chat history
        chat_history = memory.chat_memory.messages
        print(f"[chat] Chat history length: {len(chat_history)}")
        print(f"[chat] chat_history type: {type(chat_history)}, value: {chat_history}")
        
        # Generate response
        response = await chain.ainvoke({
            "context": context,
            "question": request.message,
            "chat_history": chat_history
        })
        print("[chat] Response generated")
        
        # Update memory
        memory.chat_memory.add_user_message(request.message)
        memory.chat_memory.add_ai_message(response)
        print("[chat] Memory updated")
        
        # Update session metadata
        session_metadata[session_id]["message_count"] += 1
        session_metadata[session_id]["last_activity"] = datetime.now()
        print("[chat] Session metadata updated")
        
        # Prepare sources for response
        sources = []
        for doc in docs:
            print(doc)
            source_info = {
                "content": doc.get("chunk_text", ""),
                "score": doc.get("score", None),
                "metadata": doc.get("metadata", {})
            }
            sources.append(source_info)
        print(f"[chat] Returning response with {len(sources)} sources")
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            sources=sources,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        print(f"[chat] Exception occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@chatbot_router.get("/sessions", response_model=SessionListResponse)
async def get_sessions():
    """Get list of all chat sessions."""
    sessions = []
    for session_id, metadata in session_metadata.items():
        sessions.append(SessionInfo(
            session_id=session_id,
            created_at=metadata["created_at"],
            message_count=metadata["message_count"],
            last_activity=metadata["last_activity"]
        ))
    
    return SessionListResponse(sessions=sessions)

@chatbot_router.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get chat history for a specific session."""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    memory = chat_sessions[session_id]
    messages = []
    
    for message in memory.chat_memory.messages:
        if isinstance(message, HumanMessage):
            messages.append(ChatMessage(role="user", content=message.content))
        elif isinstance(message, AIMessage):
            messages.append(ChatMessage(role="assistant", content=message.content))
        elif isinstance(message, SystemMessage):
            messages.append(ChatMessage(role="system", content=message.content))
    
    return {"session_id": session_id, "messages": messages}

@chatbot_router.delete("/sessions/{session_id}", response_model=DeleteSessionResponse)
async def delete_session(session_id: str):
    """Delete a specific chat session."""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del chat_sessions[session_id]
    del session_metadata[session_id]
    
    return DeleteSessionResponse(
        message="Session deleted successfully",
        session_id=session_id
    )

@chatbot_router.delete("/sessions")
async def clear_all_sessions():
    """Clear all chat sessions."""
    chat_sessions.clear()
    session_metadata.clear()
    
    return {"message": "All sessions cleared successfully"}

@chatbot_router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "active_sessions": len(chat_sessions)
    }

@chatbot_router.get("/providers")
async def get_available_providers():
    """Get list of available LLM providers."""
    return {
        "providers": [provider.value for provider in ModelProvider],
        "provider": get_config_value("provider"),
        "model": get_config_value("model"),
        "max_tokens": get_config_value("max_tokens"),
        "default_provider": CHATBOT_CONFIG["default_provider"],
        "default_model": CHATBOT_CONFIG["default_model"],
        "default_max_tokens": CHATBOT_CONFIG["default_max_tokens"]
    }

# Standalone FastAPI app for running this module directly
chatbot_api = FastAPI(title="Chatbot API", description="RAG-powered chatbot using LangChain", version="1.0.0")
chatbot_api.include_router(chatbot_router)