import os
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from dotenv import load_dotenv

load_dotenv(override=True)

security = HTTPBearer(auto_error=False)

def _expected_token() -> str | None:
    return os.getenv("API_TOKEN") or os.getenv("LOOMY_API_TOKEN")

def verify_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
):
    expected = _expected_token()
    if not expected:
        # No token set -> auth disabled (allow all)
        return

    provided = None
    if credentials and (credentials.scheme or "").lower() == "bearer":
        provided = credentials.credentials

    if not provided:
        provided = request.headers.get("x-api-token")
    if not provided:
        provided = request.query_params.get("token")

    if provided != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True
