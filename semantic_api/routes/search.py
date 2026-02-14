from __future__ import annotations

from fastapi import APIRouter, Depends

from semantic_api.deps import get_search_pipeline
from semantic_core.models import SearchQuery, SearchResult
from semantic_core.pipeline.searcher import SearchPipeline

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=list[SearchResult])
async def search(
    query: SearchQuery,
    pipeline: SearchPipeline = Depends(get_search_pipeline),
) -> list[SearchResult]:
    """
    Run a semantic search over indexed content.
    """
    return pipeline.search(query)
