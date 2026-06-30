from __future__ import annotations

from pathlib import Path
from typing import Optional, Set

from fastapi import HTTPException, status

from ..models import LoginRequest, LoginResponse
from ..config import Settings, get_settings

_settings = get_settings()


def _ensure_settings(settings: Settings | None) -> Settings:
    return settings or _settings


def load_keys(path: Path) -> Set[str]:
    """Load API keys from a keys file (one key per line).

    - Ignores blank lines and lines starting with '#'.
    - Keys are stripped of whitespace.
    - Missing/unreadable file → empty set.
    """
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
        return set()
    return keys


def load_auth_keys(settings: Settings | None = None) -> Set[str]:
    """Legacy flat key file (auth_keys.txt). Kept for backward compatibility."""
    settings = _ensure_settings(settings)
    return load_keys(settings.auth_keys_path)


def resolve_role(api_key: str, settings: Settings | None = None) -> Optional[str]:
    """Return the tier for a key: 'expert', 'editor', or None if unknown.

    Keys in expert_keys.txt (or the legacy auth_keys.txt) → expert.
    Keys in editor_keys.txt → editor. Expert takes precedence on overlap.
    """
    settings = _ensure_settings(settings)
    key = (api_key or "").strip()
    if not key:
        return None
    if key in load_keys(settings.expert_keys_path) or key in load_auth_keys(settings):
        return "expert"
    if key in load_keys(settings.editor_keys_path):
        return "editor"
    return None


def is_key_valid(api_key: str, settings: Settings | None = None) -> bool:
    return resolve_role(api_key, settings) is not None


def issue_token(req: LoginRequest, settings: Settings | None = None) -> LoginResponse:
    """Validate the provided API key and return a token + resolved role.

    The returned token equals the provided API key so that removing the key
    from the file immediately invalidates existing sessions.
    """
    settings = _ensure_settings(settings)
    name = req.name
    api_key = (req.api_key or "").strip()
    role = resolve_role(api_key, settings)
    if role is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_api_key")
    # Use the API key itself as the bearer token
    token = api_key
    return LoginResponse(token=token, name=name, role=role)
