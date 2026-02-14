from __future__ import annotations
import os

from fastapi import FastAPI
from dotenv import load_dotenv

from semantic_api.routes import documents, search


load_dotenv()  

app = FastAPI(title="Semantic Service API")



@app.get("/health", tags=["system"])
async def health() -> dict:
    return {"status": "ok"}


app.include_router(documents.router)
app.include_router(search.router)

