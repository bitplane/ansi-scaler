from pathlib import Path

import pytest

from ansi_scaler.manifests import read_jsonl
from ansi_scaler.runner import StageInfrastructureError, run_parallel_stage, run_stage


def parallel_processor(source: dict) -> dict:
    return {"id": f"out-{source['id']}", "parent_id": source["id"]}


def parallel_id(source: dict) -> str:
    return f"out-{source['id']}"


def test_stage_resumes_and_records_failures(tmp_path: Path) -> None:
    output = tmp_path / "output.jsonl"
    errors = tmp_path / "errors.jsonl"
    inputs = [{"id": "one"}, {"id": "bad"}, {"id": "two"}]

    def process(source: dict) -> dict:
        if source["id"] == "bad":
            raise RuntimeError("broken")
        return {"id": f"out-{source['id']}", "parent_id": source["id"]}

    def id_builder(source: dict) -> str:
        return f"out-{source['id']}"

    assert run_stage(inputs, output, errors, process, id_builder, show_progress=False) == (2, 1, 0)
    assert run_stage(inputs, output, errors, process, id_builder, show_progress=False) == (0, 0, 3)
    assert [record["id"] for record in read_jsonl(output)] == ["out-one", "out-two"]
    assert next(read_jsonl(errors))["parent_id"] == "bad"
    assert next(read_jsonl(errors))["output_id"] == "out-bad"


def test_force_rebuilds_manifest(tmp_path: Path) -> None:
    output = tmp_path / "output.jsonl"
    errors = tmp_path / "errors.jsonl"
    inputs = [{"id": "one"}]

    def processor(source: dict) -> dict:
        return {"id": "out-one", "parent_id": source["id"]}

    def id_builder(_source: dict) -> str:
        return "out-one"

    run_stage(inputs, output, errors, processor, id_builder, show_progress=False)
    assert run_stage(inputs, output, errors, processor, id_builder, force=True, show_progress=False) == (1, 0, 0)
    assert len(list(read_jsonl(output))) == 1


def test_parallel_stage_writes_manifest_in_parent(tmp_path: Path) -> None:
    output = tmp_path / "output.jsonl"
    errors = tmp_path / "errors.jsonl"
    inputs = [{"id": str(index)} for index in range(8)]

    assert run_parallel_stage(inputs, output, errors, parallel_processor, parallel_id, workers=4) == (8, 0, 0)
    assert {record["id"] for record in read_jsonl(output)} == {f"out-{index}" for index in range(8)}
    assert run_parallel_stage(inputs, output, errors, parallel_processor, parallel_id, workers=4) == (0, 0, 8)


def test_infrastructure_failure_aborts_without_poisoning_manifest(tmp_path: Path) -> None:
    output = tmp_path / "output.jsonl"
    errors = tmp_path / "errors.jsonl"
    attempted = []

    def processor(source: dict) -> dict:
        attempted.append(source["id"])
        raise StageInfrastructureError("Python.h is missing")

    with pytest.raises(StageInfrastructureError, match="Python.h"):
        run_stage(
            [{"id": "one"}, {"id": "two"}],
            output,
            errors,
            processor,
            parallel_id,
            show_progress=False,
        )

    assert attempted == ["one"]
    assert not output.exists()
    assert not errors.exists()
