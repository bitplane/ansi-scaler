from pathlib import Path

from ansi_scaler.config import load_run_config
from ansi_scaler.gc import apply_gc_plan, build_gc_plan
from ansi_scaler.locking import CorpusBusyError, corpus_lock
from ansi_scaler.manifests import read_jsonl, write_jsonl
from ansi_scaler.review.models import ReviewEvent
from ansi_scaler.stages.generate import SanaGenerator


def artifact(config, relative: str) -> str:
    path = config.artifact_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(relative.encode())
    return path.relative_to(config.data_dir).as_posix()


def gc_fixture(tmp_path: Path):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(Path("configs/runs/smoke.yaml").read_text())
    config = load_run_config(config_path)
    config.name = "gc-test"
    config.data_dir = tmp_path / "data"
    config.manifest_dir.mkdir(parents=True)
    prompt = {"id": "prompt-1", "stage": "prompts", "prompt": "crate", "seed": 1}
    current_id = SanaGenerator(config).output_id(prompt)
    old_raster = {
        "id": "old-raster",
        "parent_id": prompt["id"],
        "stage": "generate",
        "artifact": artifact(config, "rasters/00/old.png"),
    }
    current_raster = {
        "id": current_id,
        "parent_id": prompt["id"],
        "stage": "generate",
        "artifact": artifact(config, f"rasters/00/{current_id}.png"),
    }
    stale_raster = {
        "id": "stale-raster",
        "parent_id": prompt["id"],
        "stage": "generate",
        "artifact": artifact(config, "rasters/00/stale.png"),
    }
    shared_raster = {
        "id": "shared-raster",
        "parent_id": prompt["id"],
        "stage": "generate",
        "artifact": artifact(config, "rasters/00/shared.png"),
    }
    cutout = {
        "id": "old-cutout",
        "parent_id": old_raster["id"],
        "stage": "background",
        "artifact": artifact(config, "backgrounds/00/old.png"),
    }
    lod = {
        "id": "old-lod",
        "parent_id": cutout["id"],
        "stage": "lod",
        "original": cutout["artifact"],
        "levels": [
            {
                "name": "lod-0",
                "svg": artifact(config, "lod/lod-0/svg/00/old.svg"),
                "preview": artifact(config, "lod/lod-0/preview/00/old.png"),
            }
        ],
    }
    pyramid = {
        "id": "old-pyramid",
        "parent_id": lod["id"],
        "stage": "pyramid",
        "artifact": artifact(config, "pyramids/00/old.tar.zst"),
    }
    classification = {
        "id": "old-classification",
        "parent_id": cutout["id"],
        "stage": "classify",
        "artifact": cutout["artifact"],
    }
    verification = {
        "id": "old-verification",
        "parent_id": classification["id"],
        "stage": "verify",
        "artifact": cutout["artifact"],
    }
    write_jsonl(config.manifest_dir / "prompts.jsonl", [prompt])
    write_jsonl(config.manifest_dir / "rasters.jsonl", [old_raster, stale_raster, shared_raster, current_raster])
    write_jsonl(config.manifest_dir / "backgrounds.jsonl", [cutout])
    write_jsonl(config.manifest_dir / "lods.jsonl", [lod])
    write_jsonl(config.manifest_dir / "pyramids.jsonl", [pyramid])
    write_jsonl(config.manifest_dir / "classifications.jsonl", [classification])
    write_jsonl(config.manifest_dir / "verifications.jsonl", [verification])
    orphan = config.artifact_dir / "backgrounds" / "00" / "orphan.png"
    orphan.write_bytes(b"orphan")
    other = config.data_dir / "runs" / "other" / "manifests"
    other.mkdir(parents=True)
    write_jsonl(other / "rasters.jsonl", [{**shared_raster, "id": "other-reference"}])
    return config, config_path, orphan


def test_gc_keeps_complete_handover_and_cross_run_pins(tmp_path: Path) -> None:
    config, config_path, orphan = gc_fixture(tmp_path)
    plan = build_gc_plan(config, config_path)

    assert plan.removed_by_stage["generate"] == 2
    assert orphan.resolve() in plan.orphan_paths
    assert (config.artifact_dir / "rasters/00/stale.png").resolve() in plan.delete_paths
    assert (config.artifact_dir / "rasters/00/shared.png").resolve() not in plan.delete_paths
    assert (config.artifact_dir / "pyramids/00/old.tar.zst").resolve() not in plan.delete_paths
    verification_change = next(change for change in plan.changes if change.path.name == "verifications.jsonl")
    assert verification_change.after[0]["id"] == "old-verification"

    receipt = apply_gc_plan(plan)

    assert receipt.joinpath("receipt.json").is_file()
    assert receipt.joinpath("before/rasters.jsonl").is_file()
    assert not orphan.exists()
    assert not (config.artifact_dir / "rasters/00/stale.png").exists()
    assert (config.artifact_dir / "rasters/00/shared.png").exists()
    assert {record["id"] for record in read_jsonl(config.manifest_dir / "rasters.jsonl")} == {
        "old-raster",
        SanaGenerator(config).output_id(next(read_jsonl(config.manifest_dir / "prompts.jsonl"))),
    }
    second = build_gc_plan(config, config_path)
    assert second.removed_records == 0
    assert second.delete_paths == []


def test_active_review_pins_an_older_chain(tmp_path: Path) -> None:
    config, config_path, _ = gc_fixture(tmp_path)
    annotations = config.run_dir / "reviews" / "annotations.jsonl"
    annotations.parent.mkdir(parents=True)
    event = ReviewEvent(
        reviewer="tester",
        run=config.name,
        sample_id="prompt-1",
        target_stage="generate",
        target_output_id="stale-raster",
        lineage={"prompt": "prompt-1", "generate": "stale-raster"},
        outputs={"prompt": "prompt-1", "generate": "stale-raster"},
        outcome="accept",
    )
    write_jsonl(annotations, [event.model_dump(mode="json")])

    plan = build_gc_plan(config, config_path)

    assert (config.artifact_dir / "rasters/00/stale.png").resolve() not in plan.delete_paths
    assert plan.retained_reasons["active review"] == 1


def test_gc_lock_refuses_an_active_corpus_user(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    with corpus_lock(data_dir, exclusive=False):
        try:
            with corpus_lock(data_dir, exclusive=True, blocking=False):
                raise AssertionError("exclusive lock unexpectedly succeeded")
        except CorpusBusyError:
            pass


def test_gc_preserves_legacy_errors_and_drops_resolved_versioned_errors(tmp_path: Path) -> None:
    config, config_path, _ = gc_fixture(tmp_path)
    write_jsonl(
        config.manifest_dir / "rasters.errors.jsonl",
        [
            {"parent_id": "legacy", "error": "old"},
            {"parent_id": "prompt-1", "output_id": "old-raster", "error": "resolved"},
        ],
    )

    plan = build_gc_plan(config, config_path)
    change = next(change for change in plan.changes if change.path.name == "rasters.errors.jsonl")

    assert change.after == [{"parent_id": "legacy", "error": "old"}]


def test_gc_aborts_if_fingerprinted_inputs_change(tmp_path: Path) -> None:
    config, config_path, orphan = gc_fixture(tmp_path)
    plan = build_gc_plan(config, config_path)
    before = (config.manifest_dir / "rasters.jsonl").read_bytes()
    config_path.write_text(config_path.read_text() + "\n")

    try:
        apply_gc_plan(plan)
        raise AssertionError("changed plan unexpectedly applied")
    except RuntimeError as error:
        assert "changed while GC was planning" in str(error)

    assert orphan.exists()
    assert (config.manifest_dir / "rasters.jsonl").read_bytes() == before


def test_gc_reports_missing_cross_run_artifacts_without_aborting(tmp_path: Path) -> None:
    config, config_path, _ = gc_fixture(tmp_path)
    shared = config.artifact_dir / "rasters/00/shared.png"
    shared.unlink()

    plan = build_gc_plan(config, config_path)

    assert shared.resolve() in plan.missing_paths
    assert shared.resolve() not in plan.delete_paths
    assert plan.missing_by_run == {"other": 1}


def test_gc_prunes_a_broken_target_chain_and_its_descendants(tmp_path: Path) -> None:
    config, config_path, _ = gc_fixture(tmp_path)
    (config.artifact_dir / "backgrounds/00/old.png").unlink()

    plan = build_gc_plan(config, config_path)

    removed = {
        record["id"]
        for change in plan.changes
        for record in change.before
        if record not in change.after and record.get("id")
    }
    assert {"old-cutout", "old-lod", "old-pyramid", "old-classification", "old-verification"} <= removed
    assert (config.artifact_dir / "pyramids/00/old.tar.zst").resolve() in plan.delete_paths
