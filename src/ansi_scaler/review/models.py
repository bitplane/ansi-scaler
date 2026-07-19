from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


Outcome = Literal["accept", "reject", "review"]
EventType = Literal["set", "undo"]

STAGES = ("catalog", "prompt", "generate", "rembg", "lod", "pyramid", "classify", "verify")


class ReviewEvent(BaseModel):
    schema_version: Literal[1, 2] = 2
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: EventType = "set"
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reviewer: str
    run: str
    sample_id: str
    snapshot_id: str
    outputs: dict[str, str]
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
    snapshot_id: str
    outcome: Outcome
    issue_code: str | None = None
    introduced_by: str | None = None
    notes: str = ""


class UndoSubmission(BaseModel):
    event_id: str
