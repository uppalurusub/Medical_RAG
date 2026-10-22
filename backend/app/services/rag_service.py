from app.schemas.rag_schemas import QueryRequest
from app.implementations.rag_implementation import RAGImplementation
class RAGService:
    def __init__(self, impl: RAGImplementation): self.impl = impl
    def query(self, request: QueryRequest):
        q = request.question.strip()
        if not q: raise ValueError("question must not be blank")
        return self.impl.answer(q, request.top_k, request.filters, request.include_context, request.request_id)
    def query_from_eval(self, question: str, top_k: int):
        return self.impl.query_for_eval(question, top_k)
