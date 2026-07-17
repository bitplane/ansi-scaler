from pathlib import Path

import pytest

from ansi_scaler.catalog import load_catalog


def test_committed_catalog_has_five_complete_kits() -> None:
    catalog = load_catalog(Path("catalog"))
    assert set(catalog.kits) == {"woodland", "village", "city", "castle", "spaceport"}
    assert catalog.membership_count == 120
    assert all(len(kit.entries()) == 24 for kit in catalog.kits.values())


def test_unknown_concept_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "taxonomy").mkdir()
    (tmp_path / "concepts").mkdir()
    (tmp_path / "scene-kits").mkdir()
    (tmp_path / "taxonomy" / "roles.yaml").write_text("roles: [props]\n")
    (tmp_path / "concepts" / "objects.yaml").write_text("concepts:\n  - {id: known, name: Known, role: props}\n")
    contents = "\n".join(f"      - missing-{index}" for index in range(24))
    (tmp_path / "scene-kits" / "bad.yaml").write_text(f"id: bad\nname: Bad\ncontents:\n  props:\n{contents}\n")
    with pytest.raises(ValueError, match="Unknown concept"):
        load_catalog(tmp_path)
