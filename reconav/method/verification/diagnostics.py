"""Shared verification statuses and machine-auditable diagnostics."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from reconav.method.determination import Checkpoint

VerificationStatus = Literal["VALID", "WARNING", "BLOCK"]


class VerificationDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    code: str
    checkpoint: Checkpoint
    category: Literal[
        "ARTIFACT", "COMMITMENT", "GROUNDING", "LOWERING", "SOLVER", "EXECUTION"
    ]
    severity: Literal["WARNING", "BLOCK"]
    node_or_path: str
    assertion: str
    expected: Any = None
    observed: Any = None
    supporting_facts: list[str] = Field(default_factory=list)
    detail: dict[str, Any] = Field(default_factory=dict)
    affected_commitments: list[str] = Field(default_factory=list)
    repairable: bool = False
    pre_status: VerificationStatus
    post_status: VerificationStatus | None = None
    needs_backend_fix: bool


def status_for(diagnostics: list[VerificationDiagnostic]) -> VerificationStatus:
    if any(item.severity == "BLOCK" for item in diagnostics):
        return "BLOCK"
    if diagnostics:
        return "WARNING"
    return "VALID"
