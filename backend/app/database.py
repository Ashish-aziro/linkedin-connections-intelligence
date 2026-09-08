"""SQLAlchemy engine / session wiring.

Only this module knows the concrete database. Everything else goes through
``get_db`` (FastAPI dependency) or the repositories. Swapping ``DATABASE_URL``
to a Postgres DSN is the only change needed to move off SQLite.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


def _ensure_sqlite_dir(url: str) -> None:
    if url.startswith("sqlite") and ":///" in url:
        db_path = url.split(":///", 1)[1]
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def _make_engine():
    url = settings.database_url
    _ensure_sqlite_dir(url)
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args, future=True)

    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _fk_pragma(dbapi_con, _record):  # noqa: ANN001
            cur = dbapi_con.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    return engine


engine = _make_engine()
SessionLocal = sessionmaker(
    bind=engine, autoflush=False, expire_on_commit=False, future=True, class_=Session
)


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _sqlite_add_missing_columns() -> None:
    """Tiny idempotent column adder for SQLite (this project has no migration
    framework). ``create_all`` adds missing TABLES but never a new COLUMN on an
    existing table — this bridges that one gap for the columns PART 6 added to
    ``search_run_states`` on an ``app.db`` that predates them."""
    if not settings.database_url.startswith("sqlite"):
        return
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    if "search_run_states" not in insp.get_table_names():
        return
    have = {c["name"] for c in insp.get_columns("search_run_states")}
    wanted = {"search_status": "VARCHAR", "verification_metadata": "JSON"}
    missing = {k: v for k, v in wanted.items() if k not in have}
    if not missing:
        return
    with engine.begin() as conn:
        for name, sqltype in missing.items():
            conn.execute(text(f"ALTER TABLE search_run_states ADD COLUMN {name} {sqltype}"))


def init_db() -> None:
    """Create all tables. Safe to call repeatedly."""
    from app import models  # noqa: F401  — registers mappers

    Base.metadata.create_all(bind=engine)
    _sqlite_add_missing_columns()
