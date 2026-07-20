from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


Outcome = Literal["accept", "reject", "review", "unsure"]
EventType = Literal["set", "undo"]

STAGES = ("content", "prompt", "generate", "background", "lod", "pyramid", "classify", "verify")


class ReviewEvent(BaseModel):
    schema_version: Literal[1, 2, 3] = 3
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: EventType = "set"
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reviewer: str
    run: str
    sample_id: str
    snapshot_id: str = ""
    outputs: dict[str, str] = Field(default_factory=dict)
    target_stage: str | None = None
    target_output_id: str | None = None
    lineage: dict[str, str] = Field(default_factory=dict)
    outcome: Outcome | None = None
    issue_code: str | None = None
    introduced_by: str | None = None
    notes: str = ""
    supersedes: str | None = None

    @model_validator(mode="after")
    def validate_action(self) -> ReviewEvent:
        if self.event_type == "undo":
            if not self.supersedes:
                raise ValueError("Undo events must supersede an existing event")
            if self.outcome is not None:
                raise ValueError("Undo events cannot have an outcome")
            return self
        if self.outcome is None:
            raise ValueError("Review events require an outcome")
        if self.schema_version == 3:
            if self.outcome == "review":
                raise ValueError("Schema v3 uses the unsure outcome")
            if self.target_stage not in STAGES:
                raise ValueError("Schema v3 reviews require a known target stage")
            if not self.target_output_id or self.lineage.get(self.target_stage) != self.target_output_id:
                raise ValueError("Schema v3 target output must be present in its lineage")
            if self.issue_code is not None or self.introduced_by is not None:
                raise ValueError("Schema v3 reviews use the selected target stage for attribution")
            return self
        if self.outcome == "reject":
            if not self.issue_code:
                raise ValueError("Rejected reviews require an issue code")
            if self.introduced_by not in STAGES:
                raise ValueError("Rejected reviews require a known introducing stage")
        elif self.issue_code is not None or self.introduced_by is not None:
            raise ValueError("Only rejected reviews can carry an issue and stage")
        return self


class ReviewSubmission(BaseModel):
    sample_id: str
    target_stage: str
    target_output_id: str
    outcome: Literal["accept", "reject", "unsure"]
    notes: str = ""


class UndoSubmission(BaseModel):
    event_id: str
