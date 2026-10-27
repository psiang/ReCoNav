"""Strict schemas for semantic ontology grounding before instance resolution."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ASGKind = Literal["object", "region", "portal", "level"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ASGVocabulary(StrictModel):
    object: list[str]
    region: list[str]
    portal: list[str]
    level: list[str] = Field(default_factory=list)


class GroundingItem(StrictModel):
    head_text: str
    source_text: str
    asg_kind: ASGKind | None = None


class GroundingRequest(StrictModel):
    items: list[GroundingItem]
    asg_vocabulary: ASGVocabulary


class GroundingBinding(StrictModel):
    head_text: str
    asg_kind: ASGKind
    asg_label: str


class GroundingResult(StrictModel):
    bindings: list[GroundingBinding]
