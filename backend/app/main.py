from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.logging_config import configure_logging
from app.routers import datasets, enrich, health, people, search

configure_logging()
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    # TASK 5 — preload the local ML models now instead of paying their load
    # cost (sentence-transformers import + weights) inside the FIRST search
    # after every restart. Both models are independent, so they load in
    # parallel; each is internally never-raising (see embeddings.preload /
    # reranker.preload) — a failed preload just means the usual lazy load
    # happens on first use, startup itself is never blocked by it failing.
    from app.services import embeddings, reranker

    t0 = time.perf_counter()
    await asyncio.gather(
        asyncio.to_thread(embeddings.preload),
        asyncio.to_thread(reranker.preload),
    )
    log.info("model preload done in %.1fs", time.perf_counter() - t0)
    log.info(
        "startup ok — env=%s use_fixtures=%s apify=%s anthropic=%s",
        settings.environment,
        settings.use_fixtures,
        bool(settings.apify_api_token),
        bool(settings.anthropic_api_key),
    )
    yield


app = FastAPI(
    title="LinkedIn Connections Intelligence",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(datasets.router)
app.include_router(enrich.router)
app.include_router(people.router)
app.include_router(search.router)


@app.get("/")
def root() -> dict:
    return {"service": "linkedin-connections-intelligence", "docs": "/docs"}
