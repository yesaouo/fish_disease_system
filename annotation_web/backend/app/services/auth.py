from __future__ import annotations

from pathlib import Path
from typing import Set

from fastapi import HTTPException, status

from ..models import LoginRequest, LoginResponse
from ..config import Settings, get_settings

_settings = get_settings()


def _ensure_settings(settings: Settings | None) -> Settings:
    return settings or _settings


def load_auth_keys(settings: Settings | None = None) -> Set[str]:
    """Load API keys from the configured keys file (one key per line).

    - Ignores blank lines and lines starting with '#'.
    - Keys are stripped of whitespace.
    """
    settings = _ensure_settings(settings)
    path: Path = settings.auth_keys_path
    if not path.exists():
        return set()
    keys: Set[str] = set()
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = (raw or "").strip()
            if not line or line.startswith("#"):
                continue
            keys.add(line)
    except Exception:
        # If file is unreadable, treat as no keys
        return set()
    return keys


def is_key_valid(api_key: str, settings: Settings | None = None) -> bool:
    if not api_key:
        return False
    return api_key in load_auth_keys(settings)


def issue_token(req: LoginRequest, settings: Settings | None = None) -> LoginResponse:
    """Validate the provided API key and return a token.

    The returned token equals the provided API key so that removing the key
    from the file immediately invalidates existing sessions.
    """
    settings = _ensure_settings(settings)
    name = req.name
    api_key = (req.api_key or "").strip()
    if not is_key_valid(api_key, settings):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_api_key")
    # Use the API key itself as the bearer token
    token = api_key
    return LoginResponse(token=token, name=name)
