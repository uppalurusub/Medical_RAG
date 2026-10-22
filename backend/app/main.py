import logging
from fastapi import FastAPI
from app.config.config import settings
from app.routers.health_router import router as health_router
from app.routers.documents_router import router as documents_router
from app.routers.rag_router import router as rag_router
from app.routers.evaluation_router import router as evaluation_router

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = FastAPI(title=settings.app_name, version="1.0.0")
app.include_router(health_router, tags=["health"])
app.include_router(documents_router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(rag_router, prefix="/api/v1/rag", tags=["rag"])
app.include_router(evaluation_router, prefix="/api/v1/evaluation", tags=["evaluation"])
