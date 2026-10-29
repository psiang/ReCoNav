"""Strict artifacts for bounded repair semantic repair."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from reconav.method.grounding.contracts import GroundingBinding
from reconav.method.semantic.contracts import SceneIndependentSemanticContract


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RepairOutput(_StrictModel):
    """Complete semantic and label-level artifact; never scene-instance IDs."""

    model_config = ConfigDict(title="Layer4RepairOutput")

    semantic_contract: SceneIndependentSemanticContract
    grounding_bindings: list[GroundingBinding]


def repair_response_format() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "layer4_repair_output",
            "strict": True,
            "schema": RepairOutput.model_json_schema(),
        },
    }
