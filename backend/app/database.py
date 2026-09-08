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


#: SQLite columns added after their table first shipped. ``create_all`` adds
#: missing TABLES but never a new COLUMN — this bridges that one gap for a DB
#: file that predates the column (this project has no migration framework).
_LATE_COLUMNS = {
    "search_run_states": {"search_status": "VARCHAR", "verification_metadata": "JSON"},
}


def ensure_schema(eng=None) -> None:
    """Create missing tables + add late columns. Idempotent. ``eng`` defaults to
    the app engine; the eval harness passes its own engine for pilot.db / a
    copy of app.db so those get the PART 6 columns too."""
    from app import models  # noqa: F401  — registers mappers

    eng = eng or engine
    Base.metadata.create_all(bind=eng)
    if not str(eng.url).startswith("sqlite"):
        return
    from sqlalchemy import inspect, text

    insp = inspect(eng)
    tables = set(insp.get_table_names())
    for table, cols in _LATE_COLUMNS.items():
        if table not in tables:
            continue
        have = {c["name"] for c in insp.get_columns(table)}
        missing = {k: v for k, v in cols.items() if k not in have}
        if missing:
            with eng.begin() as conn:
                for name, sqltype in missing.items():
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {sqltype}"))


def init_db() -> None:
    """Create all tables + apply late columns. Safe to call repeatedly."""
    ensure_schema(engine)
