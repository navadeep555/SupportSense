"""Shared FastAPI dependencies."""
from collections.abc import Iterator

from fastapi import HTTPException, Request, status
from sqlalchemy.orm import Session

from app.services.triage import Triager


def get_db(request: Request) -> Iterator[Session]:
    """Give each request its own session and always close it afterwards.

    `yield` makes this a generator dependency: FastAPI runs the code before
    yield, injects the session into the route, and after the response runs the
    cleanup (leaving the `with` block closes the session) - even if the route raised.
    """
    factory = getattr(request.app.state, "session_factory", None)
    if factory is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database not initialised.")
    with factory() as session:
        yield session


def get_triager(request: Request) -> Triager:
    """The triager built at startup, or a rules-only one if no models were loaded."""
    return getattr(request.app.state, "triager", None) or Triager()
