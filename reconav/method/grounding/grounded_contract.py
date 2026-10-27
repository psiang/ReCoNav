"""Combine instruction contracts with occurrence-level scene bindings."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ASGBinding(StrictModel):
    asg_kind: Literal["object", "region", "portal", "level"]
    asg_label: str
    grounded_ids: list[int]


class GroundedOccurrence(StrictModel):
    position: int
    path: str
    owner_path: str
    head_text: str
    source_text: str
    asg_binding: ASGBinding


class SceneGroundedContract(StrictModel):
    """The unchanged semantic artifact plus deterministic occurrence bindings."""

    semantic_contract: dict
    occurrences: list[GroundedOccurrence]
