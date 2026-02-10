from __future__ import annotations

from fastapi import FastAPI

from semantic_api.routes import documents, search, test

app = FastAPI(title="Semantic Service API")


@app.get("/health", tags=["system"])
async def health() -> dict:
    return {"status": "ok"}


app.include_router(documents.router)
app.include_router(search.router)
app.include_router(test.router)

