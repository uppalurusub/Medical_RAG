from fastapi import APIRouter, Depends, HTTPException
from app.schemas.rag_schemas import QueryRequest, QueryResponse
from app.config.dependencies import rag_service
from app.services.rag_service import RAGService
router = APIRouter()
@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, service: RAGService = Depends(rag_service)):
    try: return service.query(request)
    except ValueError as e: raise HTTPException(400, str(e)) from e
    except RuntimeError as e: raise HTTPException(503, str(e)) from e
    except Exception as e: raise HTTPException(500, f"RAG query failed: {e}") from e
