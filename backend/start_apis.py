import uvicorn
import subprocess
import sys
import os
from pathlib import Path

def start_kb_api():
    """Start the knowledge base API server."""
    print("Starting KB API server on http://localhost:8000")
    uvicorn.run("services.kb_api:kb_api", host="0.0.0.0", port=8000, reload=True)

def start_chatbot_api():
    """Start the chatbot API server."""
    print("Starting Chatbot API server on http://localhost:8001")
    uvicorn.run("services.chatbot_api:chatbot_api", host="0.0.0.0", port=8001, reload=True)

def start_extract_api():
    """Start the extract API server."""
    print("Starting Extract API server on http://localhost:8002")
    uvicorn.run("services.extract_api:extract_api", host="0.0.0.0", port=8002, reload=True)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_type = sys.argv[1].lower()
        if api_type == "kb":
            start_kb_api()
        elif api_type == "chatbot":
            start_chatbot_api()
        elif api_type == "extract":
            start_extract_api()
        else:
            print("Usage: python start_apis.py [kb|chatbot|extract]")
            print("  kb      - Start knowledge base API on port 8000")
            print("  chatbot - Start chatbot API on port 8001")
            print("  extract - Start extract API on port 8002")
    else:
        print("Please specify which API to start:")
        print("  python start_apis.py kb      - Start knowledge base API")
        print("  python start_apis.py chatbot - Start chatbot API")
        print("  python start_apis.py extract - Start extract API")
        print("\nOr run them separately in different terminals:")
        print("  Terminal 1: uvicorn services.kb_api:kb_api --host 0.0.0.0 --port 8000 --reload")
        print("  Terminal 2: uvicorn services.chatbot_api:chatbot_api --host 0.0.0.0 --port 8001 --reload")
        print("  Terminal 3: uvicorn services.extract_api:extract_api --host 0.0.0.0 --port 8002 --reload")