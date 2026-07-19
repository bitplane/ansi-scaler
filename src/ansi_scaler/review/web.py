from __future__ import annotations

import asyncio
import hashlib
import tarfile
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated

import zstandard
from fastapi import FastAPI, HTTPException, Path as ApiPath, Query, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ansi_scaler.config import RunConfig
from ansi_scaler.review.models import STAGES, ReviewSubmission, UndoSubmission
from ansi_scaler.review.service import ReviewService


PACKAGE_ROOT = Path(__file__).parent


def _static_version() -> str:
    digest = hashlib.sha256()
    for path in sorted((PACKAGE_ROOT / "static").rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(PACKAGE_ROOT / "static").as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def create_app(config: RunConfig, *, service: ReviewService | None = None) -> FastAPI:
    owns_service = service is None
    review_service = service or ReviewService(config)

    async def refresh_loop() -> None:
        while True:
            await asyncio.sleep(config.review.refresh_seconds)
            try:
                await asyncio.to_thread(review_service.store.refresh_manifests)
            except Exception:  # noqa: BLE001 - a malformed in-progress manifest must not kill the server
                continue

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        task = asyncio.create_task(refresh_loop())
        yield
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        if owns_service:
            review_service.close()

    app = FastAPI(title="ANSI Scaler Corpus Review", lifespan=lifespan)
    app.state.review_service = review_service
    app.mount("/static", StaticFiles(directory=PACKAGE_ROOT / "static"), name="static")
    templates = Jinja2Templates(directory=PACKAGE_ROOT / "templates")
    templates.env.globals["static_version"] = _static_version()

    @app.get("/", include_in_schema=False)
    def home() -> RedirectResponse:
        return RedirectResponse("/review")

    @app.get("/review")
    def review_page(request: Request, sample: str | None = None):
        queue = review_service.queue()
        selected = review_service.sample(sample) if sample else (queue[0] if queue else None)
        if sample and selected is None:
            raise HTTPException(status_code=404, detail="Unknown sample")
        navigation = review_service.queue(include_reviewed=True) if sample else queue
        position = next(
            (index for index, item in enumerate(navigation) if selected and item["sample_id"] == selected["sample_id"]),
            -1,
        )
        return templates.TemplateResponse(
            request,
            "review.html",
            {
                "sample": selected,
                "queue_count": len(queue),
                "issues": {
                    code: [settings.label, settings.default_stage] for code, settings in config.review.issues.items()
                },
                "stages": STAGES,
                "metrics": review_service.metrics(),
                "previous_sample_id": navigation[position - 1]["sample_id"] if position > 0 else None,
                "next_sample_id": navigation[position + 1]["sample_id"]
                if 0 <= position < len(navigation) - 1
                else None,
            },
        )

    @app.get("/grid")
    def grid_page(
        request: Request,
        kit_id: str = "",
        role: str = "",
        concept_id: str = "",
        machine_decision: str = "",
        status: str = "",
        conflict: str = "",
        introduced_by: str = "",
        page: Annotated[int, Query(ge=1)] = 1,
    ):
        filters = {
            key: value
            for key, value in {
                "kit_id": kit_id,
                "role": role,
                "concept_id": concept_id,
                "machine_decision": machine_decision,
                "status": status,
                "conflict": conflict,
                "introduced_by": introduced_by,
            }.items()
            if value
        }
        all_samples = review_service.queue(filters, include_reviewed=True)
        page_size = 120
        start = (page - 1) * page_size
        samples = all_samples[start : start + page_size]
        universe = review_service.samples()
        options = {
            key: sorted({str(sample[key]) for sample in universe})
            for key in ("kit_id", "role", "concept_id", "machine_decision")
        }
        return templates.TemplateResponse(
            request,
            "grid.html",
            {
                "samples": samples,
                "total": len(all_samples),
                "page": page,
                "has_next": start + page_size < len(all_samples),
                "filters": filters,
                "options": options,
                "stages": STAGES,
            },
        )

    @app.get("/metrics")
    def metrics_page(request: Request):
        return templates.TemplateResponse(request, "metrics.html", {"metrics": review_service.metrics()})

    @app.get("/media/{record_id}")
    def media(record_id: str, level: str | None = None) -> FileResponse:
        try:
            path = review_service.media_path(record_id, level)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Unknown artifact") from error
        return FileResponse(path)

    @app.get("/api/pyramids/{record_id}/levels/{width}")
    def pyramid_level(record_id: str, width: Annotated[int, ApiPath(ge=1)]) -> dict[str, object]:
        try:
            return review_service.pyramid_level(record_id, width)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Unknown pyramid level") from error
        except (OSError, ValueError, tarfile.TarError, zstandard.ZstdError) as error:
            raise HTTPException(status_code=500, detail=f"Pyramid archive is invalid: {error}") from error

    @app.post("/api/reviews")
    def submit_review(submission: ReviewSubmission) -> dict[str, str | None]:
        try:
            event = review_service.submit(submission)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        queue = review_service.queue()
        return {
            "event_id": event.event_id,
            "next_sample_id": queue[0]["sample_id"] if queue else None,
        }

    @app.post("/api/reviews/undo")
    def undo_review(submission: UndoSubmission) -> dict[str, str]:
        try:
            event = review_service.undo(submission.event_id)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"event_id": event.event_id, "sample_id": event.sample_id}

    return app
