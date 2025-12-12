from __future__ import annotations

from fastapi import APIRouter, Depends

from ..models import LoginRequest, LoginResponse
from ..services import auth as auth_service
from .. import dependencies
from ..config import Settings

router = APIRouter(prefix="/api", tags=["auth"])


@router.post("/login", response_model=LoginResponse)
def login(req: LoginRequest, settings: Settings = Depends(dependencies.get_app_settings)) -> LoginResponse:
    return auth_service.issue_token(req, settings)
