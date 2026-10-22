from functools import lru_cache
from app.implementations.rag_implementation import RAGImplementation
from app.implementations.ingestion_implementation import IngestionImplementation
from app.implementations.evaluation_implementation import EvaluationImplementation
from app.services.rag_service import RAGService
from app.services.documents_service import DocumentsService
from app.services.evaluation_service import EvaluationService

@lru_cache
def rag_impl(): return RAGImplementation()
@lru_cache
def ingestion_impl(): return IngestionImplementation()
@lru_cache
def evaluation_impl(): return EvaluationImplementation()
def rag_service(): return RAGService(rag_impl())
def documents_service(): return DocumentsService(ingestion_impl())
def evaluation_service(): return EvaluationService(evaluation_impl(), rag_service())
