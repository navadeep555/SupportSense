"""Health-check endpoint: answers "is the backend process up and configured?"."""
from fastapi import APIRouter, Request

from app.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> dict:
    settings = get_settings()
    retriever = getattr(request.app.state, "retriever", None)
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.app_env,
        # The API can run without the index (e.g. before first ingestion);
        # this flag tells you whether retrieval will work.
        "rag_index_loaded": retriever is not None,
        "indexed_chunks": len(retriever.store.chunks) if retriever else 0,
    }
