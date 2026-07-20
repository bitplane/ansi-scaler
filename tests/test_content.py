from pathlib import Path

import pytest

from ansi_scaler.content import load_content


def test_committed_content_has_fifty_single_path_objects() -> None:
    content = load_content(Path("content"))
    assert len(content.locations) == 5
    assert len(content.objects) == 50
    assert {location.theme for location in content.locations} == {"medieval", "natural", "urban", "science-fiction"}
    assert all(len(location.objects) == 10 for location in content.locations)


def test_hierarchy_must_match_file_path(tmp_path: Path) -> None:
    path = tmp_path / "medieval" / "castle.yaml"
    path.parent.mkdir()
    path.write_text(
        "theme: science-fiction\nlocation: castle\nobjects:\n"
        "  - {id: knight, label: Knight, subject_family: knight, prompt: one knight}\n"
    )
    with pytest.raises(ValueError, match="hierarchy mismatch"):
        load_content(tmp_path)
