from __future__ import annotations

from typing import List

from fastapi import APIRouter, status


router = APIRouter(prefix="/test", tags=["test"])


@router.post(
    "/txt",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Test pipeline",
)
async def test_pipeline(text: str) -> dict:
    """
    Test the pipeline.
    """
    # Ingest docs[pdff, json, txt] 
    #  
     
    return {"status": "ok"}


