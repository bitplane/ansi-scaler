from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ansi_scaler.config import load_yaml


class Concept(BaseModel):
    id: str
    name: str
    role: str
    attributes: dict[str, list[str]] = Field(default_factory=dict)


class KitEntry(BaseModel):
    concept: str
    variant: str | None = None
    attributes: dict[str, str] = Field(default_factory=dict)


class SceneKit(BaseModel):
    id: str
    name: str
    theme: dict[str, Any] = Field(default_factory=dict)
    contents: dict[str, list[str | KitEntry]]

    def entries(self) -> list[tuple[str, KitEntry]]:
        result = []
        for role, entries in self.contents.items():
            for value in entries:
                entry = KitEntry(concept=value) if isinstance(value, str) else value
                result.append((role, entry))
        return result


class Catalog(BaseModel):
    roles: set[str]
    concepts: dict[str, Concept]
    kits: dict[str, SceneKit]

    @property
    def membership_count(self) -> int:
        return sum(len(kit.entries()) for kit in self.kits.values())


def load_catalog(root: Path) -> Catalog:
    taxonomy = load_yaml(root / "taxonomy" / "roles.yaml")
    roles = set(taxonomy.get("roles", []))

    concepts: dict[str, Concept] = {}
    for path in sorted((root / "concepts").glob("*.yaml")):
        for raw in load_yaml(path).get("concepts", []):
            concept = Concept.model_validate(raw)
            if concept.id in concepts:
                raise ValueError(f"Duplicate concept id {concept.id!r}")
            if concept.role not in roles:
                raise ValueError(f"Unknown role {concept.role!r} for concept {concept.id!r}")
            concepts[concept.id] = concept

    kits: dict[str, SceneKit] = {}
    for path in sorted((root / "scene-kits").glob("*.yaml")):
        kit = SceneKit.model_validate(load_yaml(path))
        if kit.id in kits:
            raise ValueError(f"Duplicate kit id {kit.id!r}")
        entries = kit.entries()
        if len(entries) != 24:
            raise ValueError(f"Scene kit {kit.id!r} must contain 24 memberships, found {len(entries)}")
        seen: set[str] = set()
        for role, entry in entries:
            if role not in roles:
                raise ValueError(f"Unknown role {role!r} in kit {kit.id!r}")
            if entry.concept not in concepts:
                raise ValueError(f"Unknown concept {entry.concept!r} in kit {kit.id!r}")
            membership_key = f"{role}:{entry.concept}"
            if membership_key in seen:
                raise ValueError(f"Duplicate membership {membership_key!r} in kit {kit.id!r}")
            seen.add(membership_key)
        kits[kit.id] = kit

    if not concepts:
        raise ValueError("Catalog contains no concepts")
    if not kits:
        raise ValueError("Catalog contains no scene kits")
    return Catalog(roles=roles, concepts=concepts, kits=kits)
