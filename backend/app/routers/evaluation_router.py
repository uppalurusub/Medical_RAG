from fastapi import APIRouter, Depends, HTTPException
from app.schemas.rag_schemas import EvaluationRequest, EvaluationResponse
from app.config.dependencies import evaluation_service
from app.services.evaluation_service import EvaluationService
router = APIRouter()
@router.post("/run", response_model=EvaluationResponse)
def run_evaluation(request: EvaluationRequest, service: EvaluationService = Depends(evaluation_service)):
    try: return service.run(request)
    except ValueError as e: raise HTTPException(400, str(e)) from e
    except Exception as e: raise HTTPException(500, f"Evaluation failed: {e}") from e
