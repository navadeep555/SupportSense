"""FastAPI application entry point.

Run from the backend/ folder:
    uvicorn app.main:app --reload
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import answers, health, reports, retrieval, tickets
from app.config import get_settings

# Uvicorn already configures this logger to print INFO messages to the console.
logger = logging.getLogger("uvicorn.error")
settings = get_settings()


def load_triager(retriever):
    """Load the Phase 6 models if they were trained; otherwise fall back gracefully."""
    from app.ml.classifier import TicketClassifier
    from app.ml.escalation_model import EscalationModel
    from app.services.triage import Triager

    classifier = escalation = None
    try:
        classifier = TicketClassifier.load(settings.classifier_path)
        logger.info("Ticket classifier loaded (%s)", classifier.metadata.get("model"))
    except (FileNotFoundError, RuntimeError) as exc:
        logger.warning("Ticket classifier NOT loaded: %s", exc)
    try:
        escalation = EscalationModel.load(settings.escalation_model_path)
        logger.info("Escalation model loaded (threshold %.2f)", escalation.threshold)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.warning("Escalation model NOT loaded, using keyword rules only: %s", exc)
    # Reuse the RAG embedder: same model, already in memory.
    return Triager(classifier, escalation, retriever.embedder if retriever else None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once at startup (before `yield`) and once at shutdown (after it).

    Loading the embedding model + FAISS index takes seconds, so we do it ONCE
    here instead of on every request. If the index is missing, the server still
    starts (health and other routes work) and retrieval returns 503.
    """
    from app.db.database import create_db_engine, init_db
    from app.llm.factory import create_llm_client
    from app.rag.retriever import load_retriever

    # Database: create the engine once and make sure the tables exist.
    engine = create_db_engine(settings.database_url)
    app.state.session_factory = init_db(engine)
    logger.info("Database ready: %s", engine.url.render_as_string(hide_password=True))

    app.state.llm = create_llm_client(settings)
    logger.info("LLM provider: %s (%s)", app.state.llm.provider, app.state.llm.model)

    try:
        app.state.retriever = load_retriever(settings)
        logger.info("RAG index loaded: %d chunks", len(app.state.retriever.store.chunks))
    except (FileNotFoundError, RuntimeError) as exc:
        app.state.retriever = None
        logger.warning("RAG index NOT loaded: %s", exc)

    app.state.triager = load_triager(app.state.retriever)
    yield
    engine.dispose()  # shutdown: close pooled database connections cleanly


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)

# CORS: the React app (localhost:5173) and the API (localhost:8000) are different
# "origins", so the browser blocks calls unless the API explicitly allows them.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Every route lives under /api so the frontend and backend URLs never collide.
app.include_router(health.router, prefix="/api")
app.include_router(retrieval.router, prefix="/api")
app.include_router(answers.router, prefix="/api")
app.include_router(tickets.router, prefix="/api")
app.include_router(reports.router, prefix="/api")
