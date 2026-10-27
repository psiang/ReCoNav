"""Clean repair Error request construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts/semantic_repair/system.txt"
WARNING_INTERPRETATION_POLICY = (
    "Warnings are non-blocking deterministic diagnostic signals. They may indicate "
    "a potential semantic problem, but they are not proof that the warned commitment "
    "is semantically wrong. Use the original instruction as the semantic authority "
    "when deciding whether a repair is warranted."
)


def load_repair_prompt() -> str:
    value = PROMPT_PATH.read_text(encoding="utf-8")
    if not value.strip():
        raise ValueError(f"empty repair prompt: {PROMPT_PATH}")
    return value


def build_repair_request(
    *,
    instruction: str,
    semantic_contract: dict[str, Any],
    label_grounding: list[dict[str, Any]],
    grounding_output_label_domain: dict[str, list[str]],
    error: list[dict[str, Any]],
    error_location: list[dict[str, Any]],
    previous_repair_diff: dict[str, Any] | None = None,
    previous_diagnostic: list[dict[str, Any]] | None = None,
    deterministic_warnings: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    payload: dict[str, Any] = {
        "original_instruction": instruction,
        "current_semantic_contract": semantic_contract,
        "current_asg_label_grounding": label_grounding,
        "complete_grounding_output_label_domain": grounding_output_label_domain,
        "deterministic_error": error,
        "readable_error_location": error_location,
        "previous_repair_diff": previous_repair_diff,
        "previous_diagnostic": previous_diagnostic,
        "grounding_output_order": (
            "One binding per nominal occurrence in collect_nominal_commitments traversal order "
            "of the returned semantic_contract."
        ),
    }
    if deterministic_warnings:
        payload["deterministic_warnings"] = deterministic_warnings
        payload["warning_interpretation_policy"] = WARNING_INTERPRETATION_POLICY
    return load_repair_prompt(), json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_schema_repair_request(
    *,
    instruction: str,
    semantic_source: dict[str, Any],
    schema_validation_error: Any,
    label_grounding: list[dict[str, Any]],
    grounding_output_label_domain: dict[str, list[str]],
    error: list[dict[str, Any]],
    error_location: list[dict[str, Any]],
    previous_repair_diff: dict[str, Any] | None = None,
    previous_diagnostic: list[dict[str, Any]] | None = None,
    deterministic_warnings: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    """Build the SCHEMA-mode request inside the shared repair framework."""
    system_prompt, user_prompt = build_repair_request(
        instruction=instruction,
        semantic_contract=semantic_source.get("candidate"),
        label_grounding=label_grounding,
        grounding_output_label_domain=grounding_output_label_domain,
        error=error,
        error_location=error_location,
        previous_repair_diff=previous_repair_diff,
        previous_diagnostic=previous_diagnostic,
        deterministic_warnings=deterministic_warnings,
    )
    payload = json.loads(user_prompt)
    payload.update(
        {
            "repair_mode": "SCHEMA",
            "failed_semantic_source": semantic_source,
            "schema_validation_error": schema_validation_error,
        }
    )
    return system_prompt, json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
