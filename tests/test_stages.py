from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from typer.testing import CliRunner

from ansi_scaler import cli
from ansi_scaler.config import load_run_config
from ansi_scaler.manifests import read_jsonl, write_jsonl
from ansi_scaler.stages.generate import run_generate
from ansi_scaler.stages.classify import run_classify
from ansi_scaler.stages.lod import run_lod
from ansi_scaler.stages.pyramid import (
    PYRAMID_FORMAT,
    _prepare_record_inputs,
    object_geometry,
    pyramid_id,
    source_for_width,
)
from ansi_scaler.stages.rembg import run_rembg
from ansi_scaler.stages.verify import run_verify


class FakePipeline:
    def __call__(self, **_: object) -> SimpleNamespace:
        return SimpleNamespace(images=[Image.new("RGB", (512, 512), "green")])


def test_fake_end_to_end_stages(tmp_path: Path) -> None:
    config = load_run_config(Path("configs/runs/smoke.yaml"))
    config.data_dir = tmp_path / "data"
    config.limit = 1
    config.sana.device = "cpu"
    config.manifest_dir.mkdir(parents=True)
    write_jsonl(
        config.manifest_dir / "prompts.jsonl",
        [
            {
                "id": "prompt-request",
                "stage": "prompts",
                "concept_id": "green_object",
                "concept_name": "green object",
                "prompt": "a green object",
                "negative_prompt": "background",
                "seed": 1,
            }
        ],
    )

    assert run_generate(config, pipeline=FakePipeline()) == (1, 0, 0)

    def fake_remove(image: Image.Image, **_: object) -> Image.Image:
        result = image.convert("RGBA")
        result.putalpha(Image.new("L", result.size, 255))
        return result

    assert run_rembg(config, session=object(), remove_function=fake_remove) == (1, 0, 0)
    assert run_lod(config) == (1, 0, 0)

    def fake_vlm(_: dict) -> dict:
        return {
            "message": {
                "content": """{"description":"a green object","primary_object":"object","object_count":1,
                "multiple_candidate_assets":false,"visually_coherent":true,"artifact_flags":[],"uncertainty":0.1}"""
            },
            "eval_duration": 1,
            "total_duration": 2,
        }

    assert run_classify(config, request_function=fake_vlm) == (1, 0, 0)
    classification = next(read_jsonl(config.manifest_dir / "classifications.jsonl"))
    assert classification["classification"]["primary_object"] == "object"

    def fake_llm(_: dict) -> dict:
        return {
            "message": {
                "content": """{"semantic_match":true,"cardinality_match":true,"visually_usable":true,
                "decision":"accept","rejection_reasons":[],"explanation":"The object matches.","uncertainty":0.1}"""
            },
            "eval_duration": 1,
            "total_duration": 2,
        }

    assert run_verify(config, request_function=fake_llm) == (1, 0, 0)
    verification = next(read_jsonl(config.manifest_dir / "verifications.jsonl"))
    assert verification["verification"]["decision"] == "accept"

    record = next(read_jsonl(config.manifest_dir / "lods.jsonl"))
    assert [level["name"] for level in record["levels"]] == ["lod-1", "lod-2", "lod-3"]
    assert [level["preview_size"] for level in record["levels"]] == [128, 64, 32]
    for level in record["levels"]:
        assert (config.data_dir / level["svg"]).exists()
        preview = Image.open(config.data_dir / level["preview"])
        assert preview.size == (level["preview_size"], level["preview_size"])


def test_pyramid_selects_lod_source_by_width(tmp_path: Path) -> None:
    config = load_run_config(Path("configs/runs/smoke.yaml"))
    config.data_dir = tmp_path
    record = {
        "id": "lod-record",
        "original": "original.png",
        "levels": [
            {"name": "lod-1", "svg": "lod-1.svg", "preview": "lod-1.png"},
            {"name": "lod-2", "svg": "lod-2.svg", "preview": "lod-2.png"},
            {"name": "lod-3", "svg": "lod-3.svg", "preview": "lod-3.png"},
        ],
    }

    assert source_for_width(record, 2, config) == ("lod-3", tmp_path / "lod-3.svg")
    assert source_for_width(record, 9, config) == ("lod-3", tmp_path / "lod-3.svg")
    assert source_for_width(record, 10, config) == ("lod-2", tmp_path / "lod-2.svg")
    assert source_for_width(record, 40, config) == ("lod-1", tmp_path / "lod-1.svg")
    assert source_for_width(record, 80, config) == ("original", tmp_path / "original.png")


def test_pyramid_identity_includes_renderer_version() -> None:
    config = load_run_config(Path("configs/runs/smoke.yaml"))
    record = {"id": "lod-record"}
    first = pyramid_id(record, config)
    config.chuda.version = "0.1.2"

    assert pyramid_id(record, config) != first
    assert PYRAMID_FORMAT == "ansi-scaler-pyramid-v2"


def test_pyramid_geometry_uses_alpha_bounds_and_padding(tmp_path: Path) -> None:
    config = load_run_config(Path("configs/runs/smoke.yaml"))
    config.data_dir = tmp_path
    image = Image.new("RGBA", (100, 80), (0, 0, 0, 0))
    image.paste((0, 255, 0, 255), (20, 10, 60, 50))
    image.save(tmp_path / "cutout.png")

    geometry = object_geometry({"original": "cutout.png"}, config)

    assert geometry["canvas_size"] == [100, 80]
    assert geometry["content_bbox_px"] == [20, 10, 60, 50]
    assert geometry["content_bbox"] == [0.2, 0.125, 0.6, 0.625]
    assert geometry["render_bbox_px"] == [18, 8, 62, 52]
    assert geometry["render_size_px"] == [44, 44]
    assert geometry["render_bbox"] == [0.18, 0.1, 0.62, 0.65]


def test_pyramid_prepares_identically_sized_shared_crops(tmp_path: Path) -> None:
    config = load_run_config(Path("configs/runs/smoke.yaml"))
    config.data_dir = tmp_path / "data"
    config.run_dir.mkdir(parents=True)
    original = config.data_dir / "original.png"
    original.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (100, 80), (0, 0, 0, 0))
    image.paste((0, 255, 0, 255), (20, 10, 60, 50))
    image.save(original)
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="80"><rect x="20" y="10" width="40" height="40" fill="green"/></svg>'
    levels = []
    for name in ("lod-1", "lod-2", "lod-3"):
        path = config.data_dir / f"{name}.svg"
        path.write_text(svg)
        levels.append({"name": name, "svg": path.relative_to(config.data_dir).as_posix()})
    record = {"id": "lod-record", "original": "original.png", "levels": levels}

    output_id, geometry = _prepare_record_inputs(
        record,
        config,
        str(config.run_dir),
        ("lod-3", "lod-2", "lod-1", "original"),
    )

    assert geometry["render_bbox_px"] == [18, 8, 62, 52]
    assert geometry["render_size_px"] == [44, 44]
    for source_name in ("lod-3", "lod-2", "lod-1", "original"):
        with Image.open(config.run_dir / "inputs" / source_name / f"{output_id}.png") as prepared:
            assert prepared.size == (44, 44)
            assert prepared.mode == "RGBA"


def test_pipeline_can_run_through_pyramid(tmp_path: Path, monkeypatch) -> None:
    config = load_run_config(Path("configs/runs/smoke.yaml"))
    config.data_dir = tmp_path / "data"
    called = []
    monkeypatch.setattr(cli, "load_run_config", lambda _: config)
    monkeypatch.setattr(cli, "load_catalog", lambda _: object())
    monkeypatch.setattr(cli, "write_prompt_manifest", lambda *_: called.append("prompts"))
    monkeypatch.setattr(cli, "run_generate", lambda *_args, **_kwargs: called.append("generate") or (1, 0, 0))
    monkeypatch.setattr(cli, "run_rembg", lambda *_args, **_kwargs: called.append("rembg") or (1, 0, 0))
    monkeypatch.setattr(cli, "run_lod", lambda *_args, **_kwargs: called.append("lod") or (1, 0, 0))
    monkeypatch.setattr(cli, "run_pyramid", lambda *_args, **_kwargs: called.append("pyramid") or (1, 0, 0))

    result = CliRunner().invoke(
        cli.app,
        ["run", "--run-config", "configs/runs/smoke.yaml", "--through", "pyramid"],
    )

    assert result.exit_code == 0
    assert called == ["prompts", "generate", "rembg", "lod", "pyramid"]
