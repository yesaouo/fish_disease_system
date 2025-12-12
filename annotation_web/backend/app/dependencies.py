from __future__ import annotations

from fastapi import Header, HTTPException, status

from .config import Settings, get_settings
from .services.auth import is_key_valid


def get_app_settings() -> Settings:
    return get_settings()


def require_api_key(
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> str:
    """Dependency to enforce Bearer token that matches an auth key.

    Returns the token if valid; raises 401 otherwise.
    """
    settings = get_settings()
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing_authorization")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_authorization_header")
    token = parts[1].strip()
    if not is_key_valid(token, settings):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_token")
    return token
