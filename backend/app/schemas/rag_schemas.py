from pydantic import BaseModel, Field
from typing import Any

class IngestRequest(BaseModel):
    file_paths: list[str] = Field(default_factory=list, description="Paths relative to DOCUMENTS_DIR")
    rebuild: bool = False
class IngestResponse(BaseModel):
    status: str
    files_processed: int
    pages_loaded: int
    chunks_created: int
    collection: str
    details: list[dict[str, Any]] = Field(default_factory=list)
class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=8000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    filters: dict[str, Any] | None = None
    include_context: bool = True
    request_id: str | None = None
class SourceDocument(BaseModel):
    source: str
    page: int | None = None
    chunk_id: str | None = None
    score: float | None = None
    snippet: str
class UsageInfo(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    input_cost_usd: float | None = None
    output_cost_usd: float | None = None
    total_cost_usd: float | None = None
    source: str = "unavailable"
class QueryResponse(BaseModel):
    request_id: str | None = None
    answer: str
    sources: list[SourceDocument]
    usage: UsageInfo
    latency_seconds: float
    retrieved_count: int
class EvaluationRequest(BaseModel):
    dataset_path: str = "./data/eval/rag_eval.jsonl"
    top_k: int = Field(default=4, ge=1, le=20)
    max_examples: int | None = Field(default=None, ge=1)
class EvaluationResponse(BaseModel):
    status: str
    examples_evaluated: int
    metrics: dict[str, float | int | None]
    run_id: str | None = None
    details: list[dict[str, Any]] = Field(default_factory=list)
