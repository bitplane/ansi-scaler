from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from ansi_scaler.config import load_run_config
from ansi_scaler.manifests import read_jsonl, write_jsonl
from ansi_scaler.stages.generate import run_generate
from ansi_scaler.stages.classify import run_classify
from ansi_scaler.stages.lod import run_lod
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
