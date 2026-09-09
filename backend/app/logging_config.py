"""Logging setup + secret redaction.

Never pass raw settings to the logger. Use ``redact()`` on any dict/string that
might carry a token before logging it (spec §69: never log API keys).
"""
from __future__ import annotations

import logging
import re
import sys

from app.config import settings

_SECRET_PATTERNS = [
    re.compile(r"(apify_api_[A-Za-z0-9]{6})[A-Za-z0-9]+"),
    re.compile(r"(sk-ant-[A-Za-z0-9]{6})[A-Za-z0-9_\-]+"),
]

_SECRET_KEYS = {
    "apify_api_token", "anthropic_api_key",
    "authorization", "x-api-key", "token",
}


def redact(value):
    """Return a copy of ``value`` with any recognizable secret masked."""
    if isinstance(value, str):
        out = value
        for pat in _SECRET_PATTERNS:
            out = pat.sub(r"\1…", out)
        return out
    if isinstance(value, dict):
        return {
            k: ("***" if k.lower() in _SECRET_KEYS else redact(v))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return type(value)(redact(v) for v in value)
    return value


class _RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            record.args = tuple(redact(a) for a in record.args)
        return True


def configure_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    handler.addFilter(_RedactingFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    for noisy in ("httpx", "httpcore", "apify_client", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
