from fastapi import FastAPI
from .services.kb_api import kb_api
from .services.chatbot_api import chatbot_api
from .services.extract_api import extract_api

# Create the main app
app = FastAPI()

# Mount each API under its own prefix
app.mount("/kb", kb_api)
app.mount("/chatbot", chatbot_api)
app.mount("/extract", extract_api)