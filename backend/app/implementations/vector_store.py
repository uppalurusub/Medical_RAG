from functools import lru_cache
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from app.config.config import settings

@lru_cache
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=settings.embedding_model,
        model_kwargs={"device":"cpu"}, encode_kwargs={"normalize_embeddings":True})

_store = None
def get_vector_store(force_recreate=False):
    global _store
    if force_recreate: _store = None
    if _store is None:
        settings.resolved_vector_db_path.mkdir(parents=True, exist_ok=True)
        _store = Chroma(collection_name=settings.chroma_collection,
            persist_directory=str(settings.resolved_vector_db_path), embedding_function=get_embeddings())
    return _store
