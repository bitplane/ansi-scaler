from pathlib import Path

from ansi_scaler.catalog import load_catalog
from ansi_scaler.config import load_run_config
from ansi_scaler.prompts import build_prompt_records


def test_prompt_build_is_deterministic_and_complete() -> None:
    config = load_run_config(Path("configs/runs/first.yaml"))
    catalog = load_catalog(config.catalog_dir)
    first = build_prompt_records(config, catalog)
    second = build_prompt_records(config, catalog)
    assert first == second
    assert len(first) == 1_200
    assert len({record["id"] for record in first}) == 1_200
    assert all(record["kit_id"] and record["concept_id"] and record["role"] for record in first)


def test_shared_concepts_receive_kit_context() -> None:
    config = load_run_config(Path("configs/runs/first.yaml"))
    records = build_prompt_records(config, load_catalog(config.catalog_dir))
    crate_records = [record for record in records if record["concept_id"] == "wooden_crate"]
    assert {record["kit_id"] for record in crate_records} == {"woodland", "village", "city"}
    assert len({record["prompt_id"] for record in crate_records}) == 15
