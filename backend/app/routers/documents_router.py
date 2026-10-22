from pathlib import Path
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.config.config import settings
from app.config.dependencies import documents_service
from app.services.documents_service import DocumentsService
from app.schemas.rag_schemas import IngestResponse

router = APIRouter()
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


@router.post("/ingest", response_model=IngestResponse, summary="Upload and ingest one document")
async def ingest(
    file: UploadFile = File(..., description="Browse and select one PDF or TXT file"),
    rebuild: bool = Form(False, description="Rebuild the vector collection before ingestion"),
    service: DocumentsService = Depends(documents_service),
):
    """Save one uploaded PDF/TXT file, then pass its filename to the ingestion service."""
    try:
        original_name = Path(file.filename or "").name
        suffix = Path(original_name).suffix.lower()

        if not original_name or suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="Only PDF and TXT files are supported.",
            )

        content = await file.read(MAX_FILE_SIZE_BYTES + 1)
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="The uploaded file exceeds the 25 MB limit.",
            )
        if not content:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        documents_dir = settings.resolved_documents_dir
        documents_dir.mkdir(parents=True, exist_ok=True)
        destination = documents_dir / original_name
        destination.write_bytes(content)

        # Pass a list of saved filenames, never the UploadFile object.
        return service.ingest_files([original_name], rebuild=rebuild)

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Document ingestion failed")
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {exc}",
        ) from exc
    finally:
        await file.close()
