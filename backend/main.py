from fastapi import FastAPI
from .services.kb_api import kb_api, kb_router
from .services.chatbot_api import chatbot_api, chatbot_router
from .services.extract_api import extract_api, extract_router

# Create the main app
app = FastAPI(title="Loomy Backend", description="Aggregated APIs", version="1.0.0")

# Mount each API under its own prefix
# Expose all service endpoints in the main app's OpenAPI/Swagger
app.include_router(kb_router, prefix="/kb", tags=["kb"])
app.include_router(chatbot_router, prefix="/chatbot", tags=["chatbot"])
app.include_router(extract_router, prefix="/extract", tags=["extract"])

@app.get("/")
def root():
	return {
		"name": "Loomy Backend",
		"docs": "/docs",
		"redoc": "/redoc",
		"routes": [
			"/kb/*",
			"/chatbot/*",
			"/extract/*",
		],
	}