import hashlib
import io
import json
import tarfile
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
import zstandard

from ansi_scaler.config import RunConfig, load_run_config
from ansi_scaler.identity import stable_id
from ansi_scaler.manifests import read_jsonl, write_jsonl
from ansi_scaler.review.models import ReviewSubmission
from ansi_scaler.review.service import ReviewService
from ansi_scaler.review.web import create_app
from ansi_scaler.stages.pyramid import PYRAMID_FORMAT, pyramid_id


def review_config(tmp_path: Path) -> RunConfig:
    config = load_run_config(Path("configs/runs/smoke.yaml"))
    config.name = "review-test"
    config.data_dir = tmp_path / "data"
    config.limit = None
    config.llm.prompt_version = "verifier-v1"
    config.manifest_dir.mkdir(parents=True)
    artifact = config.artifact_dir / "backgrounds" / "aa" / "cutout.png"
    artifact.parent.mkdir(parents=True)
    Image.new("RGBA", (32, 32), (10, 180, 80, 255)).save(artifact)
    raster = config.artifact_dir / "rasters" / "aa" / "raster.png"
    raster.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), "green").save(raster)
    prompt = {"id": "prompt-1", "stage": "prompts", "prompt": "green crate", "concept_id": "crate"}
    raster_id = stable_id("sana-v1", prompt["id"], config.sana.model_dump(mode="json"))
    background_id = stable_id("background-v1", raster_id, config.background.model_dump(mode="json"))
    classification_id = stable_id("vlm-classify-v1", background_id, {
        "model": config.vlm.model, "prompt_version": config.vlm.prompt_version,
        "temperature": config.vlm.temperature, "num_predict": config.vlm.num_predict,
    })
    write_jsonl(
        config.manifest_dir / "prompts.jsonl",
        [prompt],
    )
    write_jsonl(
        config.manifest_dir / "rasters.jsonl",
        [
            {
                "id": raster_id,
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
        config.manifest_dir / "backgrounds.jsonl",
        [
            {
                "id": background_id,
                "parent_id": raster_id,
                "stage": "background",
                "artifact": artifact.relative_to(config.data_dir).as_posix(),
                "background_model_sha256": config.background.sha256,
                "background_settings": config.background.model_dump(mode="json"),
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
                "id": classification_id,
                "parent_id": background_id,
                "stage": "classify",
                "artifact": artifact.relative_to(config.data_dir).as_posix(),
                "vlm_prompt_version": config.vlm.prompt_version,
                "vlm_model": "fake-vlm",
                "classification": {
                    "primary_subject": "crate",
                    "description": "one green crate",
                    "candidate_assets": ["green crate"],
                    "components": [],
                    "spatial_description": "centered",
                    "issues": [],
                    "confidence": "high",
                    "ambiguities": [],
                },
            }
        ],
    )
    write_jsonl(config.manifest_dir / "verifications.jsonl", [verification(config, classification_id, "verifier-v1", "accept")])
    return config


def verification(config: RunConfig, parent_id: str, version: str, decision: str) -> dict:
    record_id = stable_id("llm-verify-v1", parent_id, {
        "model": config.llm.model, "prompt_version": version,
        "temperature": config.llm.temperature, "num_predict": config.llm.num_predict,
    })
    return {
        "id": record_id,
        "parent_id": parent_id,
        "stage": "verify",
        "llm_prompt_version": version,
        "llm_model": "fake-llm",
        "verification": {
            "decision": decision,
            "failed_stage": "generate" if decision != "accept" else None,
            "explanation": f"machine says {decision}",
        },
    }


def add_pyramid(config: RunConfig) -> tuple[str, str]:
    cutout = next(read_jsonl(config.manifest_dir / "backgrounds.jsonl"))
    lod_id = stable_id("lod-v1", cutout["id"], config.lod.model_dump(mode="json"))
    lod_record = {"id": lod_id, "parent_id": cutout["id"], "stage": "lod", "levels": [], "original": "unused.png"}
    ansi_id = pyramid_id(lod_record, config)
    write_jsonl(
        config.manifest_dir / "lods.jsonl",
        [lod_record],
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
                "id": ansi_id,
                "parent_id": lod_id,
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
                        "source_lods": [{"name": "lod-1", "weight": 1.0}],
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "path": "levels/040.ansi",
                    }
                ],
            }
        ],
    )
    return lod_id, ansi_id


def test_review_lineage_annotation_and_metrics(tmp_path: Path) -> None:
    config = review_config(tmp_path)
    service = ReviewService(config)
    sample = service.samples()[0]
    source_before = (config.manifest_dir / "backgrounds.jsonl").read_bytes()

    event = service.submit(
        ReviewSubmission(
            sample_id=sample["sample_id"],
            target_stage="generate",
            target_output_id=sample["outputs"]["generate"],
            outcome="reject",
        )
    )

    assert service.samples()[0]["review"].event_id == event.event_id
    assert service.metrics()["matrix"] == {"unsafe_accept": 1}
    assert (config.manifest_dir / "backgrounds.jsonl").read_bytes() == source_before
    annotation = json.loads(service.store.annotation_path.read_text().strip())
    assert annotation["lineage"] == {"prompt": "prompt-1", "generate": sample["outputs"]["generate"]}
    service.close()

    reopened = ReviewService(config)
    assert reopened.samples()[0]["review"].event_id == event.event_id
    reopened.undo(event.event_id)
    assert reopened.samples()[0]["review"] is None
    reopened.close()


def test_review_uses_current_pyramid_format(tmp_path: Path) -> None:
    config = review_config(tmp_path)
    _lod_id, ansi_id = add_pyramid(config)
    manifest = config.manifest_dir / "pyramids.jsonl"
    current = next(read_jsonl(manifest))
    legacy = {**current, "id": "pyramid-v1", "pyramid_format": "ansi-scaler-pyramid-v1"}
    write_jsonl(manifest, [current, legacy])

    service = ReviewService(config)

    assert service.samples()[0]["pyramid"]["id"] == ansi_id
    service.close()


def test_review_normalises_legacy_single_lod_provenance(tmp_path: Path) -> None:
    config = review_config(tmp_path)
    _lod_id, ansi_id = add_pyramid(config)
    manifest = config.manifest_dir / "pyramids.jsonl"
    record = next(read_jsonl(manifest))
    record["pyramid_levels"][0].pop("source_lods")
    write_jsonl(manifest, [record])

    service = ReviewService(config)

    assert service.pyramid_level(ansi_id, 40)["source_lods"] == [{"name": "lod-1", "weight": 1.0}]
    service.close()


def test_changed_model_conflict_is_prioritised(tmp_path: Path) -> None:
    config = review_config(tmp_path)
    service = ReviewService(config)
    old = service.samples()[0]
    service.submit(
        ReviewSubmission(
            sample_id=old["sample_id"],
            target_stage="generate",
            target_output_id=old["outputs"]["generate"],
            outcome="reject",
        )
    )
    write_jsonl(
        config.manifest_dir / "verifications.jsonl",
        [verification(config, old["outputs"]["classify"], "verifier-v1", "reject"), verification(config, old["outputs"]["classify"], "verifier-v2", "accept")],
    )
    config.llm.prompt_version = "verifier-v2"
    service.store.refresh_manifests()

    queued = service.queue(include_reviewed=True)

    assert queued[0]["verification"]["llm_prompt_version"] == "verifier-v2"
    assert queued[0]["conflict"] is True
    service.close()


def test_new_snapshot_review_supersedes_prior_asset_review_without_resurrection(tmp_path: Path) -> None:
    config = review_config(tmp_path)
    service = ReviewService(config)
    old = service.samples()[0]
    first = service.submit(
        ReviewSubmission(
            sample_id=old["sample_id"],
            target_stage="verify",
            target_output_id=old["outputs"]["verify"],
            outcome="accept",
        )
    )
    write_jsonl(
        config.manifest_dir / "verifications.jsonl",
        [verification(config, old["outputs"]["classify"], "verifier-v1", "accept"), verification(config, old["outputs"]["classify"], "verifier-v2", "accept")],
    )
    config.llm.prompt_version = "verifier-v2"
    service.store.refresh_manifests()
    current = service.samples()[0]

    second = service.submit(
        ReviewSubmission(
            sample_id=current["sample_id"],
            target_stage="verify",
            target_output_id=current["outputs"]["verify"],
            outcome="accept",
        )
    )

    assert second.supersedes == first.event_id
    assert [event.event_id for event in service.store.active_reviews()] == [second.event_id]
    service.undo(second.event_id)
    assert service.store.active_reviews() == []
    service.close()


def test_accepting_machine_failure_advances_stage_by_stage(tmp_path: Path) -> None:
    config = review_config(tmp_path)
    initial_service = ReviewService(config)
    initial = initial_service.samples()[0]
    initial_service.close()
    write_jsonl(config.manifest_dir / "verifications.jsonl", [verification(config, initial["outputs"]["classify"], "verifier-v1", "reject")])
    service = ReviewService(config)
    sample = service.samples()[0]

    assert sample["focus_stage"] == "verify"
    service.submit(
        ReviewSubmission(
            sample_id=sample["sample_id"],
            target_stage="verify",
            target_output_id=sample["outputs"]["verify"],
            outcome="accept",
        )
    )

    advanced = service.samples()[0]
    assert advanced["focus_stage"] == "verify"
    assert advanced["review_complete"] is True
    assert advanced["conflict"] is True
    service.close()


def test_stage_review_survives_later_change_but_not_ancestor_change(tmp_path: Path) -> None:
    config = review_config(tmp_path)
    service = ReviewService(config)
    sample = service.samples()[0]
    event = service.submit(
        ReviewSubmission(
            sample_id=sample["sample_id"],
            target_stage="background",
            target_output_id=sample["outputs"]["background"],
            outcome="unsure",
        )
    )

    write_jsonl(
        config.manifest_dir / "verifications.jsonl",
        [verification(config, sample["outputs"]["classify"], "verifier-v1", "accept"), verification(config, sample["outputs"]["classify"], "verifier-v2", "accept")],
    )
    config.llm.prompt_version = "verifier-v2"
    service.store.refresh_manifests()
    assert service.samples()[0]["reviews"]["background"].event_id == event.event_id

    background = next(read_jsonl(config.manifest_dir / "backgrounds.jsonl"))
    write_jsonl(config.manifest_dir / "backgrounds.jsonl", [{**background, "id": "cutout-2"}])
    service.store.refresh_manifests()
    changed = service.samples()[0]
    assert "background" not in changed["reviews"]
    assert changed["stale_reviews"]["background"].event_id == event.event_id
    assert changed["focus_stage"] == "generate"
    assert changed["review_complete"] is False
    service.close()


def test_reviews_for_different_stages_coexist(tmp_path: Path) -> None:
    config = review_config(tmp_path)
    service = ReviewService(config)
    sample = service.samples()[0]
    for stage in ("generate", "background"):
        service.submit(
            ReviewSubmission(
                sample_id=sample["sample_id"],
                target_stage=stage,
                target_output_id=sample["outputs"][stage],
                outcome="accept",
            )
        )

    assert {event.target_stage for event in service.store.active_reviews()} == {"generate", "background"}
    assert set(service.samples()[0]["reviews"]) == {"generate", "background"}
    service.close()


def test_current_classifier_error_is_a_reviewable_failed_stage(tmp_path: Path) -> None:
    config = review_config(tmp_path)
    classification = next(read_jsonl(config.manifest_dir / "classifications.jsonl"))
    (config.manifest_dir / "classifications.jsonl").unlink()
    (config.manifest_dir / "verifications.jsonl").unlink()
    write_jsonl(
        config.manifest_dir / "classifications.errors.jsonl",
        [{
            "parent_id": classification["parent_id"],
            "output_id": classification["id"],
            "error_type": "OllamaStructuredOutputError",
            "error": "VLM response was incomplete",
            "traceback": "trace",
            "diagnostics": {"partial_content": "repeated output"},
        }],
    )
    service = ReviewService(config)
    sample = service.samples()[0]

    assert sample["focus_stage"] == "classify"
    assert [tab["stage"] for tab in sample["stage_tabs"]] == [
        "generate", "background", "classify", "verify", "lod", "pyramid"
    ]
    assert next(tab for tab in sample["stage_tabs"] if tab["stage"] == "classify")["status"] == "failed"
    assert next(tab for tab in sample["stage_tabs"] if tab["stage"] == "verify")["disabled"] is True

    event = service.submit(ReviewSubmission(
        sample_id=sample["sample_id"], target_stage="classify",
        target_output_id=classification["id"], outcome="reject", notes="runaway",
    ))
    assert event.lineage["classify"] == classification["id"]
    assert event.notes == "runaway"
    service.close()


def test_review_web_routes_and_safe_media(tmp_path: Path) -> None:
    config = review_config(tmp_path)
    _lod_id, ansi_id = add_pyramid(config)
    service = ReviewService(config)
    app = create_app(config, service=service)
    sample = service.samples()[0]
    with TestClient(app) as client:
        review_response = client.get("/review")
        assert review_response.status_code == 200
        assert 'id="ansi-stage" class="stage-tab status-ready selected"' in review_response.text
        assert 'id="ansi-width" type="range" min="40" max="40" value="40"' in review_response.text
        assert "review.css?v=" in review_response.text
        assert "review.js?v=" in review_response.text
        ansi_response = client.get(f"/api/pyramids/{ansi_id}/levels/40")
        assert ansi_response.status_code == 200
        assert ansi_response.json()["palette"] == [[10, 20, 30], [40, 50, 60]]
        assert ansi_response.json()["runs"][0] == ["▃" + (" " * 39), 0, 1]
        assert ansi_response.json()["source_lods"] == [{"name": "lod-1", "weight": 1.0}]
        assert "html" not in ansi_response.json()
        assert client.get(f"/api/pyramids/{ansi_id}/levels/41").status_code == 404
        assert client.get(f"/api/pyramids/{sample['outputs']['background']}/levels/40").status_code == 404
        assert "wooden crate" in client.get("/grid").text
        assert client.get("/metrics").status_code == 200
        assert client.get(f"/media/{sample['outputs']['background']}").headers["content-type"] == "image/png"
        assert client.get("/media/../../etc/passwd").status_code == 404
        response = client.post(
            "/api/reviews",
            json={
                "sample_id": sample["sample_id"],
                "target_stage": "pyramid",
                "target_output_id": sample["outputs"]["pyramid"],
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
        assert 'id="ansi-stage"' in response.text
        assert 'id="ansi-stage" class="stage-tab status-pending"' in response.text
    service.close()


def test_review_falls_back_to_generated_raster_without_background(tmp_path: Path) -> None:
    config = review_config(tmp_path)
    for name in ("backgrounds.jsonl", "lods.jsonl", "pyramids.jsonl", "classifications.jsonl", "verifications.jsonl"):
        (config.manifest_dir / name).unlink(missing_ok=True)
    service = ReviewService(config)
    sample = service.samples()[0]
    assert sample["cutout"] is None
    with TestClient(create_app(config, service=service)) as client:
        review = client.get("/review")
        assert review.status_code == 200
        assert f'data-image="/media/{sample["outputs"]["generate"]}"' in review.text
        assert f'src="/media/{sample["outputs"]["generate"]}"' in review.text
        grid = client.get("/grid")
        assert grid.status_code == 200
        assert f'src="/media/{sample["outputs"]["generate"]}"' in grid.text
    service.close()


def test_review_defaults_to_latest_available_stage_without_ansi(tmp_path: Path) -> None:
    config = review_config(tmp_path)
    preview = config.artifact_dir / "lods" / "aa" / "lod-0.png"
    preview.parent.mkdir(parents=True)
    Image.new("RGBA", (32, 32), (20, 40, 180, 255)).save(preview)
    cutout = next(read_jsonl(config.manifest_dir / "backgrounds.jsonl"))
    lod_id = stable_id("lod-v1", cutout["id"], config.lod.model_dump(mode="json"))
    write_jsonl(
        config.manifest_dir / "lods.jsonl",
        [
            {
                "id": lod_id,
                "parent_id": cutout["id"],
                "stage": "lod",
                "levels": [{"name": "lod-0", "preview": preview.relative_to(config.data_dir).as_posix()}],
            }
        ],
    )
    service = ReviewService(config)
    with TestClient(create_app(config, service=service)) as client:
        review = client.get("/review")
        assert review.status_code == 200
        assert 'id="lod-stage" class="stage-tab status-ready selected"' in review.text
        assert f'src="/media/{lod_id}?level=lod-0"' in review.text
        grid = client.get("/grid")
        assert f'src="/media/{lod_id}?level=lod-0"' in grid.text
    service.close()
