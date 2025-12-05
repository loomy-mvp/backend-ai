from fastapi import FastAPI
from contextlib import asynccontextmanager
from .services.kb_api import kb_router
from .services.chatbot_api import chatbot_router
from .services.write_api import write_router
from .utils.db_utils import DBUtils

@asynccontextmanager
async def lifespan(app: FastAPI):
	"""Lifespan context manager for startup and shutdown events."""
	# Startup
	try:
		DBUtils.initialize_pool()
		print("[startup] Database connection pool initialized successfully")
	except Exception as e:
		print(f"[startup] Warning: Failed to initialize database pool: {e}")
		print("[startup] Application will continue, but database operations may fail")
	
	yield
	
	# Shutdown
	try:
		DBUtils.close_pool()
		print("[shutdown] Database connection pool closed successfully")
	except Exception as e:
		print(f"[shutdown] Warning: Error closing database pool: {e}")

# Create the main app
app = FastAPI(
	title="Loomy Backend",
	description="Aggregated APIs",
	version="1.0.0",
	lifespan=lifespan
)

# Mount each API under its own prefix
# Expose all service endpoints in the main app's OpenAPI/Swagger
app.include_router(kb_router, prefix="/kb", tags=["kb"])
app.include_router(chatbot_router, prefix="/chatbot", tags=["chatbot"])
app.include_router(write_router, prefix="/write", tags=["write"])

@app.get("/")
def root():
	return {
		"name": "Loomy AI Backend",
		"docs": "/docs",
		"redoc": "/redoc",
		"routes": [
			"/kb/*",
			"/chatbot/*",
			"/write/*"
		],
	}