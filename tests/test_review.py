import hashlib
import io
import json
import tarfile
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
import zstandard

from ansi_scaler.config import RunConfig, load_run_config
from ansi_scaler.manifests import read_jsonl, write_jsonl
from ansi_scaler.review.models import ReviewSubmission
from ansi_scaler.review.service import ReviewService
from ansi_scaler.review.web import create_app
from ansi_scaler.stages.pyramid import PYRAMID_FORMAT


def review_config(tmp_path: Path) -> RunConfig:
    config = load_run_config(Path("configs/runs/smoke.yaml"))
    config.name = "review-test"
    config.data_dir = tmp_path / "data"
    config.limit = None
    config.llm.prompt_version = "verifier-v1"
    config.manifest_dir.mkdir(parents=True)
    artifact = config.artifact_dir / "cutouts" / "aa" / "cutout.png"
    artifact.parent.mkdir(parents=True)
    Image.new("RGBA", (32, 32), (10, 180, 80, 255)).save(artifact)
    raster = config.artifact_dir / "rasters" / "aa" / "raster.png"
    raster.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), "green").save(raster)
    write_jsonl(
        config.manifest_dir / "prompts.jsonl",
        [{"id": "prompt-1", "stage": "prompts", "prompt": "green crate", "concept_id": "crate"}],
    )
    write_jsonl(
        config.manifest_dir / "rasters.jsonl",
        [
            {
                "id": "raster-1",
                "parent_id": "prompt-1",
                "stage": "generate",
                "artifact": raster.relative_to(config.data_dir).as_posix(),
                "prompt": "green crate",
                "concept_id": "crate",
                "concept_name": "wooden crate",
                "kit_id": "village",
                "role": "props",
            }
        ],
    )
    write_jsonl(
        config.manifest_dir / "cutouts.jsonl",
        [
            {
                "id": "cutout-1",
                "parent_id": "raster-1",
                "stage": "rembg",
                "artifact": artifact.relative_to(config.data_dir).as_posix(),
                "rembg_model_sha256": config.rembg.sha256,
                "concept_id": "crate",
                "concept_name": "wooden crate",
                "kit_id": "village",
                "role": "props",
            }
        ],
    )
    write_jsonl(
        config.manifest_dir / "classifications.jsonl",
        [
            {
                "id": "classification-1",
                "parent_id": "cutout-1",
                "stage": "classify",
                "artifact": artifact.relative_to(config.data_dir).as_posix(),
                "vlm_prompt_version": config.vlm.prompt_version,
                "vlm_model": "fake-vlm",
                "classification": {
                    "primary_object": "crate",
                    "description": "one green crate",
                    "object_count": 1,
                    "uncertainty": 0.0,
                },
            }
        ],
    )
    write_jsonl(config.manifest_dir / "verifications.jsonl", [verification("verify-1", "verifier-v1", "accept")])
    return config


def verification(record_id: str, version: str, decision: str) -> dict:
    return {
        "id": record_id,
        "parent_id": "classification-1",
        "stage": "verify",
        "llm_prompt_version": version,
        "llm_model": "fake-llm",
        "verification": {"decision": decision, "explanation": f"machine says {decision}"},
    }


def add_pyramid(config: RunConfig) -> None:
    write_jsonl(
        config.manifest_dir / "lods.jsonl",
        [{"id": "lod-1", "parent_id": "cutout-1", "stage": "lod", "levels": [], "original": "unused.png"}],
    )
    data = b"\x1b[38;2;10;20;30;48;2;40;50;60m\xe2\x96\x83" + (b" " * 39) + b"\x1b[0m\n"
    archive_path = config.artifact_dir / "pyramids" / "aa" / "pyramid.tar.zst"
    archive_path.parent.mkdir(parents=True)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        info = tarfile.TarInfo("levels/040.ansi")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    archive_path.write_bytes(zstandard.ZstdCompressor().compress(buffer.getvalue()))
    write_jsonl(
        config.manifest_dir / "pyramids.jsonl",
        [
            {
                "id": "pyramid-1",
                "parent_id": "lod-1",
                "stage": "pyramid",
                "artifact": archive_path.relative_to(config.data_dir).as_posix(),
                "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                "chuda_version": config.chuda.version,
                "pyramid_format": PYRAMID_FORMAT,
                "pyramid_levels": [
                    {
                        "width": 40,
                        "rows": 1,
                        "source_lod": "lod-1",
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "path": "levels/040.ansi",
                    }
                ],
            }
        ],
    )


def test_review_lineage_annotation_and_metrics(tmp_path: Path) -> None:
    config = review_config(tmp_path)
    service = ReviewService(config)
    sample = service.samples()[0]
    source_before = (config.manifest_dir / "cutouts.jsonl").read_bytes()

    event = service.submit(
        ReviewSubmission(
            sample_id=sample["sample_id"],
            snapshot_id=sample["snapshot_id"],
            outcome="reject",
            issue_code="wrong_subject",
            introduced_by="generate",
        )
    )

    assert service.samples()[0]["review"].event_id == event.event_id
    assert service.metrics()["matrix"] == {"unsafe_accept": 1}
    assert (config.manifest_dir / "cutouts.jsonl").read_bytes() == source_before
    annotation = json.loads(service.store.annotation_path.read_text().strip())
    assert annotation["outputs"]["verify"] == "verify-1"
    service.close()

    reopened = ReviewService(config)
    assert reopened.samples()[0]["review"].event_id == event.event_id
    reopened.undo(event.event_id)
    assert reopened.samples()[0]["review"] is None
    reopened.close()


def test_review_uses_current_pyramid_format(tmp_path: Path) -> None:
    config = review_config(tmp_path)
    add_pyramid(config)
    manifest = config.manifest_dir / "pyramids.jsonl"
    current = next(read_jsonl(manifest))
    legacy = {**current, "id": "pyramid-v1", "pyramid_format": "ansi-scaler-pyramid-v1"}
    write_jsonl(manifest, [current, legacy])

    service = ReviewService(config)

    assert service.samples()[0]["pyramid"]["id"] == "pyramid-1"
    service.close()


def test_changed_model_conflict_is_prioritised(tmp_path: Path) -> None:
    config = review_config(tmp_path)
    service = ReviewService(config)
    old = service.samples()[0]
    service.submit(
        ReviewSubmission(
            sample_id=old["sample_id"],
            snapshot_id=old["snapshot_id"],
            outcome="reject",
            issue_code="wrong_subject",
            introduced_by="generate",
        )
    )
    write_jsonl(
        config.manifest_dir / "verifications.jsonl",
        [verification("verify-1", "verifier-v1", "reject"), verification("verify-2", "verifier-v2", "accept")],
    )
    config.llm.prompt_version = "verifier-v2"
    service.store.refresh_manifests()

    queued = service.queue()

    assert queued[0]["outputs"]["verify"] == "verify-2"
    assert queued[0]["conflict"] is True
    service.close()


def test_review_web_routes_and_safe_media(tmp_path: Path) -> None:
    config = review_config(tmp_path)
    add_pyramid(config)
    service = ReviewService(config)
    app = create_app(config, service=service)
    sample = service.samples()[0]
    with TestClient(app) as client:
        review_response = client.get("/review")
        assert review_response.status_code == 200
        assert 'id="ansi-stage" class="selected"' in review_response.text
        assert 'id="ansi-width" type="range" min="40" max="40" value="40"' in review_response.text
        assert "review.css?v=" in review_response.text
        assert "review.js?v=" in review_response.text
        ansi_response = client.get("/api/pyramids/pyramid-1/levels/40")
        assert ansi_response.status_code == 200
        assert ansi_response.json()["palette"] == [[10, 20, 30], [40, 50, 60]]
        assert ansi_response.json()["runs"][0] == ["▃" + (" " * 39), 0, 1]
        assert "html" not in ansi_response.json()
        assert client.get("/api/pyramids/pyramid-1/levels/41").status_code == 404
        assert client.get("/api/pyramids/cutout-1/levels/40").status_code == 404
        assert "wooden crate" in client.get("/grid").text
        assert client.get("/metrics").status_code == 200
        assert client.get("/media/cutout-1").headers["content-type"] == "image/png"
        assert client.get("/media/../../etc/passwd").status_code == 404
        response = client.post(
            "/api/reviews",
            json={
                "sample_id": sample["sample_id"],
                "snapshot_id": sample["snapshot_id"],
                "outcome": "accept",
            },
        )
        assert response.status_code == 200
        assert response.json()["next_sample_id"] is None
    service.close()


def test_review_falls_back_to_cutout_without_pyramid(tmp_path: Path) -> None:
    config = review_config(tmp_path)
    service = ReviewService(config)
    with TestClient(create_app(config, service=service)) as client:
        response = client.get("/review")
        assert response.status_code == 200
        assert 'id="ansi-stage"' not in response.text
        assert 'class="selected" data-image="/media/cutout-1"' in response.text
    service.close()
