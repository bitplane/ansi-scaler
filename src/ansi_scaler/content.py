from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ansi_scaler.config import load_yaml


class ObjectSpecification(BaseModel):
    id: str
    label: str
    subject_family: str
    prompt: str
    exclusions: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class ContentLocation(BaseModel):
    theme: str
    location: str
    objects: list[ObjectSpecification]
    source: Path


class ContentLibrary(BaseModel):
    locations: list[ContentLocation]

    @property
    def objects(self) -> list[tuple[ContentLocation, ObjectSpecification]]:
        return [(location, item) for location in self.locations for item in location.objects]


def load_content(root: Path) -> ContentLibrary:
    locations: list[ContentLocation] = []
    object_ids: set[str] = set()
    paths = sorted(root.glob("*/*.yaml"))
    for path in paths:
        payload = load_yaml(path)
        location = ContentLocation.model_validate({**payload, "source": path})
        expected_theme = path.parent.name
        expected_location = path.stem
        if location.theme != expected_theme or location.location != expected_location:
            raise ValueError(f"Content hierarchy mismatch in {path}: expected {expected_theme}/{expected_location}")
        if not location.objects:
            raise ValueError(f"Content location contains no objects: {path}")
        local_ids: set[str] = set()
        for item in location.objects:
            if item.id in local_ids:
                raise ValueError(f"Duplicate object id {item.id!r} in {path}")
            local_ids.add(item.id)
            canonical_id = f"{location.theme}/{location.location}/{item.id}"
            if canonical_id in object_ids:
                raise ValueError(f"Duplicate canonical object id {canonical_id!r}")
            object_ids.add(canonical_id)
        locations.append(location)
    if not locations:
        raise ValueError(f"Content library contains no theme/location files: {root}")
    return ContentLibrary(locations=locations)
