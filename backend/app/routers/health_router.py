from fastapi import APIRouter
from app.config.config import settings
router = APIRouter()
@router.get("/health")
def health():
    return {"status":"ok","app":settings.app_name,"model_configured":bool(settings.resolved_model),
            "api_key_configured":bool(settings.resolved_api_key)}
