from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

# Point at a throwaway SQLite file BEFORE anything imports app.config / app.database.
_TEST_DB = Path(tempfile.gettempdir()) / f"lci_test_{uuid.uuid4().hex}.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB.as_posix()}"
os.environ["USE_FIXTURES"] = "true"
os.environ["ENVIRONMENT"] = "development"
os.environ["EMBEDDINGS_ENABLED"] = "false"
os.environ["SEMANTIC_ENABLED"] = "false"
os.environ["LLM_QUERY_INTERPRETATION"] = "false"
os.environ["LLM_REASON_GENERATION"] = "false"
# V4 PART 5 — the final audit is an LLM path; off by default in tests (like the
# reason generator). Audit tests enable it and mock final_auditor._audit_batch.
os.environ["FINAL_RESULT_AUDIT_ENABLED"] = "false"
os.environ["DEVELOPMENT_BATCH_SIZE"] = "3"
os.environ.setdefault("GROQ_API_KEY", "")
os.environ.setdefault("APIFY_API_TOKEN", "")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.database import Base, SessionLocal, engine, init_db  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_db():
    Base.metadata.drop_all(bind=engine)
    init_db()
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def db():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    try:
        _TEST_DB.unlink(missing_ok=True)
    except OSError:
        pass


SAMPLE_CSV = b"""Notes:
"When exporting your connection data, ..."

First Name,Last Name,URL,Email Address,Company,Position,Connected On
Jane,Smith,https://www.linkedin.com/in/jane-smith,jane@example.com,Google,Senior Software Engineer,01 Jan 2025
John,Doe,https://linkedin.com/in/john-doe/?trk=abc,,Amazon,SDE,15 Mar 2024
Dup,Jane,https://www.linkedin.com/in/jane-smith?utm_source=x,,Google,SWE,02 Feb 2025
NoUrl,Person,,,SomeCo,Manager,03 Mar 2023
"""
