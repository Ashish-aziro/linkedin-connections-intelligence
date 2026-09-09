from __future__ import annotations

import logging
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
    log.info(
        "startup ok — env=%s use_fixtures=%s apify=%s anthropic=%s groq=%s openrouter=%s",
        settings.environment,
        settings.use_fixtures,
        bool(settings.apify_api_token),
        bool(settings.anthropic_api_key),
        bool(settings.groq_api_key),
        bool(settings.openrouter_api_key),
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
