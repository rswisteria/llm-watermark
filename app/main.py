from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from app.config import ServiceSettings
from app.schemas import DetectRequest, DetectionResponse, GenerateRequest, HealthResponse
from app.services import (
    DetectionService,
    GenerationEvent,
    GenerationQueueFullError,
    GenerationService,
    ModelNotReadyError,
)

logger = logging.getLogger(__name__)
STATIC_INDEX = Path(__file__).parent / "static" / "index.html"


def _sse(event: GenerationEvent) -> str:
    return f"event: {event.kind}\ndata: {json.dumps(event.payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def create_app(
    settings: ServiceSettings | None = None,
    generation_service: GenerationService | None = None,
    detection_service: DetectionService | None = None,
) -> FastAPI:
    resolved_settings = settings or ServiceSettings.from_env()
    config = resolved_settings.watermark_config()
    generation = generation_service or GenerationService(resolved_settings, config)
    detection = detection_service or getattr(generation, "detection", None)
    if detection is None:
        detection = DetectionService(None, config)
    loading_started = False

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal loading_started
        if not loading_started:
            generation.start_loading()
            loading_started = True
        yield

    app = FastAPI(lifespan=lifespan)

    class InvalidWatermarkOverride(ValueError):
        pass

    def _resolve_watermark(override):
        try:
            return resolved_settings.watermark_config(override)
        except ValueError:
            raise InvalidWatermarkOverride() from None

    @app.exception_handler(InvalidWatermarkOverride)
    async def invalid_watermark_handler(request: Request, exc: InvalidWatermarkOverride):
        return JSONResponse(status_code=400, content={"detail": "invalid request"})

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=400, content={"detail": "invalid request"})

    @app.exception_handler(ModelNotReadyError)
    async def model_not_ready_handler(request: Request, exc: ModelNotReadyError):
        return JSONResponse(status_code=503, content={"detail": "model is not ready"})

    @app.exception_handler(GenerationQueueFullError)
    async def queue_full_handler(request: Request, exc: GenerationQueueFullError):
        return JSONResponse(status_code=429, content={"detail": "generation queue is full"})

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        logger.error("unexpected API error: %s", type(exc).__name__)
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    @app.post("/api/detect", response_model=DetectionResponse)
    def detect(payload: DetectRequest):
        result = detection.classify(
            payload.text,
            include_tokens=payload.include_tokens,
            config=_resolve_watermark(payload.watermark),
        )
        body = result.model_dump(mode="json")
        if result.tokens is None:
            # Drop the top-level key only; response_model_exclude_none would also
            # strip legitimate nested nulls such as TokenDetail.green for the
            # first token (no previous token to form a bigram with).
            body.pop("tokens", None)
        return JSONResponse(content=body)

    @app.post("/api/generate")
    def generate(payload: GenerateRequest):
        iterator = iter(generation.begin(
            payload.prompt,
            resolved_settings.max_tokens(payload.max_new_tokens),
            payload.seed,
            _resolve_watermark(payload.watermark),
            inspect=payload.inspect,
        ))
        try:
            first = next(iterator)
        except StopIteration:
            first = None

        def stream() -> Iterator[str]:
            if first is not None:
                yield _sse(first)
            for event in iterator:
                yield _sse(event)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/api/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            model_loaded=generation.health(),
            model_name=resolved_settings.model_name,
            gamma=resolved_settings.gamma,
            delta=resolved_settings.delta,
            z_threshold=resolved_settings.z_threshold,
        )

    @app.get("/")
    async def index():
        return FileResponse(STATIC_INDEX)

    return app


app = create_app()
