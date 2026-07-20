from pathlib import Path

from ansi_scaler.config import load_run_config
from ansi_scaler.content import load_content
from ansi_scaler.prompts import build_prompt_records


def test_prompt_build_is_deterministic_and_exactly_one_hundred() -> None:
    config = load_run_config(Path("configs/runs/first.yaml"))
    content = load_content(config.content_dir)
    first = build_prompt_records(config, content)
    second = build_prompt_records(config, content)
    assert first == second
    assert len(first) == 100
    assert len({record["id"] for record in first}) == 100
    assert len({record["prompt_family_id"] for record in first}) == 50
    assert all(record["theme"] and record["location"] and record["specification_id"] for record in first)


def test_categories_are_metadata_not_implicit_prompt_fragments() -> None:
    config = load_run_config(Path("configs/runs/first.yaml"))
    records = build_prompt_records(config, load_content(config.content_dir))
    knight = next(record for record in records if record["object_id"] == "standing-knight")
    assert knight["theme"] == "medieval"
    assert knight["location"] == "castle"
    assert "castle" not in knight["semantic_prompt"].lower()
    assert "castle" not in knight["prompt"].lower()
    assert knight["prompt"].count(config.sana.presentation_prompt) == 1
