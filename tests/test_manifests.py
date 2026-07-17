from pathlib import Path

from ansi_scaler.manifests import read_jsonl
from ansi_scaler.runner import run_stage


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

    assert run_stage(inputs, output, errors, process, id_builder) == (2, 1, 0)
    assert run_stage(inputs, output, errors, process, id_builder) == (0, 0, 3)
    assert [record["id"] for record in read_jsonl(output)] == ["out-one", "out-two"]
    assert next(read_jsonl(errors))["parent_id"] == "bad"


def test_force_rebuilds_manifest(tmp_path: Path) -> None:
    output = tmp_path / "output.jsonl"
    errors = tmp_path / "errors.jsonl"
    inputs = [{"id": "one"}]

    def processor(source: dict) -> dict:
        return {"id": "out-one", "parent_id": source["id"]}

    def id_builder(_source: dict) -> str:
        return "out-one"

    run_stage(inputs, output, errors, processor, id_builder)
    assert run_stage(inputs, output, errors, processor, id_builder, force=True) == (1, 0, 0)
    assert len(list(read_jsonl(output))) == 1
