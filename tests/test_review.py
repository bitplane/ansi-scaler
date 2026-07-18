import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from ansi_scaler.config import RunConfig, load_run_config
from ansi_scaler.manifests import write_jsonl
from ansi_scaler.review.models import ReviewSubmission
from ansi_scaler.review.service import ReviewService
from ansi_scaler.review.web import create_app


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
    service = ReviewService(config)
    app = create_app(config, service=service)
    sample = service.samples()[0]
    with TestClient(app) as client:
        assert client.get("/review").status_code == 200
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
