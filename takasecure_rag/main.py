from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from openai import APIConnectionError, APIStatusError, AuthenticationError
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .graph import AdaptiveRAG
from .schemas import ChatRequest, ChatResponse


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = AdaptiveRAG(get_settings())
    yield


app = FastAPI(
    title="TakaSecure Adaptive RAG",
    version="0.1.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request):
    try:
        return request.app.state.rag.invoke(payload)
    except AuthenticationError as error:
        logger.error("Model gateway rejected the configured API credential.")
        raise HTTPException(
            status_code=502,
            detail="The model gateway rejected its configured credential. Update VLLM_API_KEY.",
        ) from error
    except APIConnectionError as error:
        logger.error("Model gateway is unreachable: %s", error)
        raise HTTPException(
            status_code=503,
            detail="The model gateway is currently unreachable.",
        ) from error
    except APIStatusError as error:
        logger.error("Model gateway returned HTTP %s.", error.status_code)
        raise HTTPException(
            status_code=502,
            detail=f"The model gateway returned HTTP {error.status_code}.",
        ) from error
    except Exception as error:
        logger.exception("Policy request failed.")
        raise HTTPException(
            status_code=500,
            detail="The policy pipeline could not complete the request.",
        ) from error
