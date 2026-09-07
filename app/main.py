"""
Nyaya, the ASGI entry point.

Run locally:
    uvicorn app.main:app --reload

On Render the start command is:
    uvicorn app.main:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import router as api_router
from .core import config, embeddings
from .core.store import get_index

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("nyaya")

WEB_DIR = Path(__file__).resolve().parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the index up front so the first request is not the one that pays for it
    index = get_index()
    if index.ready:
        log.info("Corpus loaded: %s chunks", index.n_chunks)
    else:
        log.error("Corpus unavailable: %s", index.error)

    # The embedding model takes a few seconds to load, so warm it on a background
    # thread. The server starts accepting traffic immediately and Render's health
    # check passes rather than timing out on a cold boot.
    threading.Thread(target=embeddings.warm_up, name="embed-warmup", daemon=True).start()
    yield


app = FastAPI(
    title=f"{config.APP_NAME} API",
    description="Citation grounded retrieval over a corpus of Indian law.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
)

app.include_router(api_router)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong on the server. Please try again."},
    )


app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/favicon.svg", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(WEB_DIR / "static" / "favicon.svg")
