from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from pydantic import ValidationError
from typer.testing import CliRunner

from ansi_scaler import cli
from ansi_scaler.active import active_backgrounds, active_rasters
from ansi_scaler.config import load_run_config
from ansi_scaler.config import BackgroundSettings
from ansi_scaler.manifests import read_jsonl, write_jsonl
from ansi_scaler.stages.generate import run_generate
from ansi_scaler.stages.classify import Classification, run_classify
from ansi_scaler.stages.lod import run_lod
from ansi_scaler.stages.pyramid import (
    INPUT_RASTERIZATION_CONTRACT,
    LOD_BLEND_RADIUS,
    PYRAMID_FORMAT,
    _blend_rgba_buffers,
    _crop_record_sources,
    _prepare_record,
    object_geometry,
    pyramid_id,
    pyramid_queue_limits,
    pyramid_worker_count,
    run_pyramid,
    source_for_width,
)
from ansi_scaler.stages.background import run_background
from ansi_scaler.stages.background import BackgroundProcessor
from ansi_scaler.stages.generate import SanaGenerator
from ansi_scaler.stages.verify import run_verify
from ansi_scaler.runner import StageInfrastructureError


class FakePipeline:
    def __call__(self, **_: object) -> SimpleNamespace:
        return SimpleNamespace(images=[Image.new("RGB", (512, 512), "green")])


def test_cli_loads_pwd_dotenv_without_overriding_shell_environment(tmp_path: Path, monkeypatch) -> None:
    run_config = Path("configs/runs/smoke.yaml").resolve()
    (tmp_path / ".env").write_text("OLLAMA_HOST=http://remote-ollama:11434\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OLLAMA_HOST", "http://shell-ollama:11434")

    shell_config = cli._config(run_config)
    assert shell_config.vlm.endpoint == "http://shell-ollama:11434"
    assert shell_config.llm.endpoint == "http://shell-ollama:11434"

    monkeypatch.delenv("OLLAMA_HOST")
    dotenv_config = cli._config(run_config)
    assert dotenv_config.vlm.endpoint == "http://remote-ollama:11434"
    assert dotenv_config.llm.endpoint == "http://remote-ollama:11434"


def test_active_lineage_excludes_superseded_rasters_and_backgrounds(tmp_path: Path) -> None:
    config = load_run_config(Path("configs/runs/smoke.yaml"))
    config.data_dir = tmp_path / "data"
    config.manifest_dir.mkdir(parents=True)
    prompt = {"id": "current-prompt", "stage": "prompts"}
    current_raster_id = SanaGenerator(config).output_id(prompt)
    current_raster = {"id": current_raster_id, "parent_id": prompt["id"], "stage": "generate"}
    stale_raster = {"id": "stale-raster", "parent_id": "old-prompt", "stage": "generate"}
    current_background_id = BackgroundProcessor(config).output_id(current_raster)
    current_background = {
        "id": current_background_id,
        "parent_id": current_raster_id,
        "stage": "background",
    }
    stale_background = {"id": "stale-background", "parent_id": "stale-raster", "stage": "background"}
    write_jsonl(config.manifest_dir / "prompts.jsonl", [prompt])
    write_jsonl(config.manifest_dir / "rasters.jsonl", [stale_raster, current_raster])
    write_jsonl(config.manifest_dir / "backgrounds.jsonl", [stale_background, current_background])

    assert [record["id"] for record in active_rasters(config)] == [current_raster_id]
    assert [record["id"] for record in active_backgrounds(config)] == [current_background_id]


def test_background_provider_settings_are_provider_specific() -> None:
    defaults = BackgroundSettings()
    assert defaults.provider == "lucida-transformers"
    assert defaults.model == "egeorcun/lucida"
    assert defaults.revision == "28632b8fefc5431cfc1e42ed9d6123d785ea49ad"

    lucida = BackgroundSettings(provider="lucida-transformers", model="egeorcun/lucida", revision="pinned-revision")
    assert lucida.sha256 is None
    assert lucida.model_path is None
    with pytest.raises(ValidationError, match="pinned revision"):
        BackgroundSettings(provider="lucida-transformers", model="egeorcun/lucida", revision=None)


def test_fake_end_to_end_stages(tmp_path: Path) -> None:
    config = load_run_config(Path("configs/runs/smoke.yaml"))
    config.data_dir = tmp_path / "data"
    config.limit = 1
    config.sana.device = "cpu"
    config.background = BackgroundSettings(
        provider="rembg-onnx",
        model="birefnet-general",
        revision=None,
        sha256="58f621f00f5d756097615970a88a791584600dcf7c45b18a0a6267535a1ebd3c",
        model_path=Path("~/.u2net/birefnet-general.onnx"),
    )
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

    assert run_background(config, session=object(), remove_function=fake_remove) == (1, 0, 0)
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
    assert [level["name"] for level in record["levels"]] == ["lod-0", "lod-1", "lod-2", "lod-3"]
    assert [level["preview_size"] for level in record["levels"]] == [512, 512, 512, 512]
    for level in record["levels"]:
        assert (config.data_dir / level["svg"]).exists()
        preview = Image.open(config.data_dir / level["preview"])
        assert preview.size == (level["preview_size"], level["preview_size"])


def test_lucida_background_provider_uses_model_alpha(tmp_path: Path) -> None:
    config = load_run_config(Path("configs/runs/smoke.yaml"))
    config.data_dir = tmp_path / "data"
    config.limit = None
    config.background.provider = "lucida-transformers"
    config.background.model = "egeorcun/lucida"
    config.background.revision = "pinned-revision"
    config.background.device = "cpu"
    config.background.dtype = "float32"
    config.background.input_size = 64
    config.background.sha256 = None
    config.background.model_path = None
    source = config.data_dir / "source.png"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (24, 16), "red").save(source)
    config.manifest_dir.mkdir(parents=True)
    write_jsonl(config.manifest_dir / "rasters.jsonl", [{"id": "raster", "artifact": "source.png"}])

    class FakeLucida:
        def __call__(self, tensor: torch.Tensor) -> list[torch.Tensor]:
            return [torch.zeros((tensor.shape[0], 1, tensor.shape[2], tensor.shape[3]))]

    assert run_background(config, model=FakeLucida()) == (1, 0, 0)
    record = next(read_jsonl(config.manifest_dir / "backgrounds.jsonl"))
    assert record["background_provider"] == "lucida-transformers"
    assert record["background_model_revision"] == "pinned-revision"
    with Image.open(config.data_dir / record["artifact"]) as result:
        assert result.size == (24, 16)
        assert result.getchannel("A").getextrema() == (127, 127)


def test_classify_continues_after_ollama_retries_are_exhausted(tmp_path: Path) -> None:
    config = load_run_config(Path("configs/runs/smoke.yaml"))
    config.data_dir = tmp_path / "data"
    config.limit = None
    config.vlm.retry_attempts = 4
    config.vlm.retry_initial_seconds = 0
    config.manifest_dir.mkdir(parents=True)
    records = []
    for index in range(2):
        artifact = config.data_dir / f"cutout-{index}.png"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (16, 16), (20, 40, 60, 255)).save(artifact)
        records.append({"id": f"cutout-{index}", "stage": "background", "artifact": artifact.name})
    write_jsonl(config.manifest_dir / "backgrounds.jsonl", records)
    calls = 0

    def flaky_vlm(_: dict) -> dict:
        nonlocal calls
        calls += 1
        if calls <= config.vlm.retry_attempts:
            raise ConnectionError("Ollama restarted")
        return {
            "message": {
                "content": Classification(
                    description="object",
                    primary_object="object",
                    object_count=1,
                    multiple_candidate_assets=False,
                    visually_coherent=True,
                    artifact_flags=[],
                    uncertainty=0.0,
                ).model_dump_json()
            }
        }

    assert run_classify(config, request_function=flaky_vlm) == (1, 1, 0)
    assert calls == 5
    assert [item["parent_id"] for item in read_jsonl(config.manifest_dir / "classifications.errors.jsonl")] == [
        "cutout-0"
    ]
    assert [item["parent_id"] for item in read_jsonl(config.manifest_dir / "classifications.jsonl")] == ["cutout-1"]


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
            {"name": "lod-0", "svg": "lod-0.svg", "preview": "lod-0.png"},
        ],
    }

    def weights(width: int) -> list[tuple[str, float]]:
        return [(name, weight) for name, _path, weight in source_for_width(record, width, config)]

    assert LOD_BLEND_RADIUS == 4
    assert weights(2) == [("lod-3", 1.0)]
    assert weights(6) == [("lod-3", 1.0)]
    assert weights(7) == [("lod-3", 0.875), ("lod-2", 0.125)]
    assert weights(8) == [("lod-3", 0.75), ("lod-2", 0.25)]
    assert weights(10) == [("lod-3", 0.5), ("lod-2", 0.5)]
    assert weights(13) == [("lod-3", 0.125), ("lod-2", 0.875)]
    assert weights(14) == [("lod-2", 1.0)]
    assert weights(36) == [("lod-2", 1.0)]
    assert weights(40) == [("lod-2", 0.5), ("lod-1", 0.5)]
    assert weights(44) == [("lod-1", 1.0)]
    assert weights(76) == [("lod-1", 1.0)]
    assert weights(80) == [("lod-1", 0.5), ("lod-0", 0.5)]
    assert weights(84) == [("lod-0", 1.0)]
    assert weights(120) == [("lod-0", 1.0)]
    for lower, higher, boundary in (("lod-3", "lod-2", 10), ("lod-2", "lod-1", 40), ("lod-1", "lod-0", 80)):
        for offset in range(-LOD_BLEND_RADIUS + 1, LOD_BLEND_RADIUS):
            higher_weight = (offset + LOD_BLEND_RADIUS) / (2 * LOD_BLEND_RADIUS)
            assert weights(boundary + offset) == [
                (lower, 1.0 - higher_weight),
                (higher, higher_weight),
            ]


def test_pyramid_blends_premultiplied_rgba_without_transparent_colour_leak() -> None:
    lower = (1, 1, bytes((0, 0, 255, 0)))
    higher = (1, 1, bytes((255, 0, 0, 255)))

    width, height, data = _blend_rgba_buffers(lower, higher, 0.5)

    assert (width, height) == (1, 1)
    assert tuple(data) == (255, 0, 0, 127)


def test_pyramid_identity_includes_renderer_version() -> None:
    config = load_run_config(Path("configs/runs/smoke.yaml"))
    record = {"id": "lod-record"}
    first = pyramid_id(record, config)
    config.chuda.version = "0.1.2"

    assert pyramid_id(record, config) != first
    assert PYRAMID_FORMAT == "ansi-scaler-pyramid-v3"
    assert INPUT_RASTERIZATION_CONTRACT == "shared-crop-premultiplied-lod-blend-v2"


def test_pyramid_identity_does_not_depend_on_backend() -> None:
    config = load_run_config(Path("configs/runs/smoke.yaml"))
    record = {"id": "lod-record"}
    first = pyramid_id(record, config)
    config.chuda.backend = "cpu"

    assert pyramid_id(record, config) == first


def test_pyramid_worker_count_reserves_cpu_and_memory_headroom(monkeypatch) -> None:
    config = load_run_config(Path("configs/runs/smoke.yaml"))
    gib = 1024**3
    monkeypatch.setattr("ansi_scaler.stages.pyramid.psutil.cpu_count", lambda logical: 8)
    monkeypatch.setattr(
        "ansi_scaler.stages.pyramid.psutil.virtual_memory",
        lambda: SimpleNamespace(total=10 * gib, available=3 * gib),
    )

    assert pyramid_worker_count(config) == 6
    assert pyramid_queue_limits(6) == (7, 2)

    config.resources.pyramid_workers = 20
    assert pyramid_worker_count(config) == 8


def test_pyramid_worker_count_keeps_one_worker_under_memory_pressure(monkeypatch) -> None:
    config = load_run_config(Path("configs/runs/smoke.yaml"))
    gib = 1024**3
    monkeypatch.setattr("ansi_scaler.stages.pyramid.psutil.cpu_count", lambda logical: 16)
    monkeypatch.setattr(
        "ansi_scaler.stages.pyramid.psutil.virtual_memory",
        lambda: SimpleNamespace(total=10 * gib, available=1 * gib),
    )

    assert pyramid_worker_count(config) == 1


def test_pyramid_rejects_overlapping_lod_blend_windows(tmp_path: Path) -> None:
    config = load_run_config(Path("configs/runs/smoke.yaml"))
    config.data_dir = tmp_path / "data"
    config.chuda.lod_2_below = config.chuda.lod_3_below + (2 * LOD_BLEND_RADIUS) - 1

    with pytest.raises(ValueError, match="too close"):
        run_pyramid(config)


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
    for name in ("lod-0", "lod-1", "lod-2", "lod-3"):
        path = config.data_dir / f"{name}.svg"
        path.write_text(svg)
        levels.append({"name": name, "svg": path.relative_to(config.data_dir).as_posix()})
    record = {"id": "lod-record", "original": "original.png", "levels": levels}

    geometry = object_geometry(record, config)
    sources = _crop_record_sources(record, config, geometry, ("lod-3", "lod-2", "lod-1", "lod-0"))

    assert geometry["render_bbox_px"] == [18, 8, 62, 52]
    assert geometry["render_size_px"] == [44, 44]
    for source in sources.values():
        assert (source.width, source.height) == (44, 44)

    prepared_geometry, buffers, _elapsed = _prepare_record(record, config, (8, 10, 12, 20))
    assert prepared_geometry == geometry
    assert {"blend-008", "blend-010", "blend-012", "lod-2"} <= buffers.keys()
    assert "blend-020" not in buffers
    assert {(width, height) for width, height, _data in buffers.values()} == {(44, 44)}


def test_pyramid_renders_and_packs_one_record_without_staging_tree(tmp_path: Path) -> None:
    config = load_run_config(Path("configs/runs/smoke.yaml"))
    config.data_dir = tmp_path / "data"
    config.chuda.backend = "cpu"
    config.chuda.min_width = 8
    config.chuda.max_width = 12
    config.manifest_dir.mkdir(parents=True)
    original = config.data_dir / "cutout.png"
    original.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (16, 16), (20, 40, 60, 255)).save(original)
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"><rect width="16" height="16" fill="#14283c"/></svg>'
    levels = []
    for name in ("lod-0", "lod-1", "lod-2", "lod-3"):
        path = config.data_dir / f"{name}.svg"
        path.write_text(svg)
        levels.append({"name": name, "svg": path.relative_to(config.data_dir).as_posix()})
    write_jsonl(
        config.manifest_dir / "lods.jsonl",
        [{"id": "lod-record", "original": "cutout.png", "levels": levels}],
    )

    assert run_pyramid(config) == (1, 0, 0)

    record = next(read_jsonl(config.manifest_dir / "pyramids.jsonl"))
    assert [level["width"] for level in record["pyramid_levels"]] == [8, 9, 10, 11, 12]
    assert record["pyramid_levels"][0]["source_lods"] == [
        {"name": "lod-3", "weight": 0.75},
        {"name": "lod-2", "weight": 0.25},
    ]
    assert record["pyramid_levels"][2]["source_lods"] == [
        {"name": "lod-3", "weight": 0.5},
        {"name": "lod-2", "weight": 0.5},
    ]
    assert record["chuda_backends"] == ["cpu"]
    assert (config.data_dir / record["artifact"]).is_file()
    assert not list(config.run_dir.glob("ansi-scaler-pyramids-*"))
    assert run_pyramid(config) == (0, 0, 1)


def test_pyramid_continues_after_malformed_prepared_record(tmp_path: Path) -> None:
    config = load_run_config(Path("configs/runs/smoke.yaml"))
    config.data_dir = tmp_path / "data"
    config.chuda.backend = "cpu"
    config.chuda.min_width = 2
    config.chuda.max_width = 3
    config.resources.pyramid_workers = 2
    config.manifest_dir.mkdir(parents=True)
    records = []
    for record_id, color in (("empty", (0, 0, 0, 0)), ("valid", (20, 40, 60, 255))):
        original = config.data_dir / f"{record_id}.png"
        original.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (16, 16), color).save(original)
        levels = []
        for name in ("lod-0", "lod-1", "lod-2", "lod-3"):
            path = config.data_dir / f"{record_id}-{name}.svg"
            path.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16">'
                '<rect width="16" height="16" fill="#14283c"/></svg>'
            )
            levels.append({"name": name, "svg": path.relative_to(config.data_dir).as_posix()})
        records.append(
            {
                "id": record_id,
                "original": original.relative_to(config.data_dir).as_posix(),
                "levels": levels,
            }
        )
    write_jsonl(config.manifest_dir / "lods.jsonl", records)

    assert run_pyramid(config) == (1, 1, 0)
    assert [record["parent_id"] for record in read_jsonl(config.manifest_dir / "pyramids.jsonl")] == ["valid"]
    assert [record["parent_id"] for record in read_jsonl(config.manifest_dir / "pyramids.errors.jsonl")] == ["empty"]


def test_pyramid_archive_failure_is_infrastructure_error(tmp_path: Path, monkeypatch) -> None:
    config = load_run_config(Path("configs/runs/smoke.yaml"))
    config.data_dir = tmp_path / "data"
    config.chuda.backend = "cpu"
    config.chuda.min_width = 2
    config.chuda.max_width = 3
    config.resources.pyramid_workers = 1
    config.manifest_dir.mkdir(parents=True)
    original = config.data_dir / "cutout.png"
    original.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (16, 16), (20, 40, 60, 255)).save(original)
    levels = []
    for name in ("lod-0", "lod-1", "lod-2", "lod-3"):
        path = config.data_dir / f"{name}.svg"
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16">'
            '<rect width="16" height="16" fill="#14283c"/></svg>'
        )
        levels.append({"name": name, "svg": path.relative_to(config.data_dir).as_posix()})
    write_jsonl(
        config.manifest_dir / "lods.jsonl",
        [{"id": "lod-record", "original": "cutout.png", "levels": levels}],
    )

    def fail_archive(*_args, **_kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr("ansi_scaler.stages.pyramid._pack_archive", fail_archive)

    with pytest.raises(StageInfrastructureError, match="archive infrastructure failed"):
        run_pyramid(config)
    assert not list(read_jsonl(config.manifest_dir / "pyramids.jsonl"))


def test_pipeline_can_run_through_pyramid(tmp_path: Path, monkeypatch) -> None:
    config = load_run_config(Path("configs/runs/smoke.yaml"))
    config.data_dir = tmp_path / "data"
    called = []
    monkeypatch.setattr(cli, "load_run_config", lambda _: config)
    monkeypatch.setattr(cli, "load_content", lambda _: object())
    monkeypatch.setattr(cli, "write_prompt_manifest", lambda *_: called.append("prompts"))
    monkeypatch.setattr(cli, "run_generate", lambda *_args, **_kwargs: called.append("generate") or (1, 0, 0))
    monkeypatch.setattr(cli, "run_background", lambda *_args, **_kwargs: called.append("background") or (0, 1, 0))
    monkeypatch.setattr(cli, "run_lod", lambda *_args, **_kwargs: called.append("lod") or (1, 0, 0))
    monkeypatch.setattr(cli, "run_pyramid", lambda *_args, **_kwargs: called.append("pyramid") or (1, 0, 0))

    result = CliRunner().invoke(
        cli.app,
        ["run", "--run-config", "configs/runs/smoke.yaml", "--through", "pyramid"],
    )

    assert result.exit_code == 0
    assert called == ["prompts", "generate", "background", "lod", "pyramid"]


def test_pipeline_can_run_through_verification(tmp_path: Path, monkeypatch) -> None:
    config = load_run_config(Path("configs/runs/smoke.yaml"))
    config.data_dir = tmp_path / "data"
    called = []
    monkeypatch.setattr(cli, "load_run_config", lambda _: config)
    monkeypatch.setattr(cli, "load_content", lambda _: object())
    monkeypatch.setattr(cli, "write_prompt_manifest", lambda *_: called.append("prompts"))
    monkeypatch.setattr(cli, "run_generate", lambda *_args, **_kwargs: called.append("generate") or (1, 0, 0))
    monkeypatch.setattr(cli, "run_background", lambda *_args, **_kwargs: called.append("background") or (1, 0, 0))
    monkeypatch.setattr(cli, "run_lod", lambda *_args, **_kwargs: called.append("lod") or (1, 0, 0))
    monkeypatch.setattr(cli, "run_pyramid", lambda *_args, **_kwargs: called.append("pyramid") or (1, 0, 0))
    monkeypatch.setattr(cli, "run_classify", lambda *_args, **_kwargs: called.append("classify") or (1, 0, 0))
    monkeypatch.setattr(cli, "run_verify", lambda *_args, **_kwargs: called.append("verify") or (1, 0, 0))

    result = CliRunner().invoke(
        cli.app,
        ["run", "--run-config", "configs/runs/smoke.yaml", "--through", "verify"],
    )

    assert result.exit_code == 0
    assert called == ["prompts", "generate", "background", "lod", "pyramid", "classify", "verify"]
