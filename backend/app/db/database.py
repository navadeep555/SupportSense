"""Database connection: the engine (how to reach the DB) and sessions (one unit of work).

Engine  - created ONCE per app. Holds a pool of connections to the database.
Session - created per request. Tracks the objects you load/add and turns them
          into SQL INSERT/UPDATE statements when you call commit().
"""
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    """Parent class of every table model. Base.metadata knows all tables."""


def create_db_engine(url: str) -> Engine:
    if not url.startswith("sqlite"):
        # pre_ping: test a pooled connection before using it, so a database
        # restart doesn't surface as a random error on the next request.
        return create_engine(url, pool_pre_ping=True)

    # FastAPI runs normal `def` routes in a thread pool, but Python's sqlite3
    # driver refuses (by default) a connection used from a different thread.
    kwargs = {"connect_args": {"check_same_thread": False}}
    if url in ("sqlite://", "sqlite:///:memory:"):
        # In-memory DB (tests): each new connection would get a brand-new EMPTY
        # database, so StaticPool makes everyone share one connection.
        kwargs["poolclass"] = StaticPool
    engine = create_engine(url, **kwargs)

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_conn, _record):
        # SQLite ignores FOREIGN KEY rules unless this is switched on per connection.
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    return engine


def init_db(engine: Engine) -> sessionmaker:
    """Create missing tables and return a factory that makes sessions.

    create_all only ADDS missing tables; it never alters existing ones. Changing
    a column later needs a migration tool (Alembic) - planned for Phase 9.
    """
    from app.db import models  # noqa: F401  (importing registers the tables on Base)

    Base.metadata.create_all(engine)
    # expire_on_commit=False: objects stay readable after commit, so a route
    # can return them without SQLAlchemy re-loading them from the database.
    return sessionmaker(bind=engine, expire_on_commit=False)
