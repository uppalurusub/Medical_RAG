from app.schemas.rag_schemas import EvaluationRequest
from app.implementations.evaluation_implementation import EvaluationImplementation
from app.services.rag_service import RAGService
class EvaluationService:
    def __init__(self, impl: EvaluationImplementation, rag: RAGService):
        self.impl, self.rag = impl, rag
    def run(self, request: EvaluationRequest):
        return self.impl.evaluate(request.dataset_path, request.top_k, request.max_examples,
                                  lambda q, k: self.rag.query_from_eval(q, k))
