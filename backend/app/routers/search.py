from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import repositories as repo
from app.database import get_db
from app.schemas import SearchRequest, SearchResponse

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def run_search(payload: SearchRequest, db: Session = Depends(get_db)) -> SearchResponse:
    if not repo.get_dataset(db, payload.dataset_id):
        raise HTTPException(404, "dataset not found")
    from app.services.full_verification import VerificationIncompleteError
    from app.services.search_service import run_connection_search

    try:
        resp = run_connection_search(db, dataset_id=payload.dataset_id, query=payload.query)
    except VerificationIncompleteError as e:
        # FULL SONNET VERIFICATION could not complete for every filtered candidate.
        # Nothing was persisted — the whole search is a retryable failure, never a
        # partial/unverified result set.
        db.rollback()
        raise HTTPException(
            status_code=503,
            detail={
                "error": "verification_incomplete",
                "message": "Full Sonnet verification could not be completed. Please retry.",
                "retryable": True,
                "verification": e.metadata,
            },
        ) from e
    db.commit()
    return resp


@router.get("/search/{search_id}", response_model=SearchResponse)
def get_search(search_id: str, db: Session = Depends(get_db)) -> SearchResponse:
    from app.services.search_service import load_search

    resp = load_search(db, search_id)
    if not resp:
        raise HTTPException(404, "search not found")
    return resp


@router.get("/datasets/{dataset_id}/searches")
def search_history(dataset_id: str, db: Session = Depends(get_db)) -> list[dict]:
    if not repo.get_dataset(db, dataset_id):
        raise HTTPException(404, "dataset not found")
    return [
        {
            "search_id": s.id,
            "query": s.query_text,
            "created_at": s.created_at.isoformat(),
            "total_candidates": s.total_candidates,
            "llm_provider": s.llm_provider,
        }
        for s in repo.list_search_queries(db, dataset_id)
    ]
