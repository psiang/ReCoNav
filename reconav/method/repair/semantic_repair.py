"""Bounded same-model repair semantic repair and deterministic re-forwarding."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from reconav.scene.asg import ASG
from reconav.method.verification.contract_index import build_contract_index
from reconav.method.verification.verify import VerificationOutcome, run_verification
from reconav.method.grounding.contracts import GroundingBinding
from reconav.method.grounding.label_grounding import (
    collect_nominal_commitments,
    determine_groundings,
    validate_bindings,
    vocabulary_from_entity_payload,
)
from reconav.method.repair.diagnostics import (
    build_error,
    build_error_location,
    build_grounding_output_domain_audit,
    build_warnings,
    complete_asg_label_vocabulary,
    repair_eligibility,
)
from reconav.method.repair.models import (
    RepairOutput,
    repair_response_format,
)
from reconav.method.repair.prompting import (
    build_repair_request,
    build_schema_repair_request,
)
from reconav.method.semantic.contracts import SceneIndependentSemanticContract

RepairCall = Callable[[str, str, dict[str, Any]], tuple[str, dict[str, Any]]]
RepairPreflight = Callable[[str, str, dict[str, Any]], dict[str, Any]]
_STALE_KEYS = {
    "scene_binding",
    "lowering",
    "execution",
    "predicted",
    "route",
    "stage_verification",
    "failure",
}
_FORBIDDEN_OUTPUT_KEYS = {
    "asg_id",
    "asg_ids",
    "grounded_id",
    "grounded_ids",
    "coordinates",
    "waypoints",
    "route_result",
}


@dataclass(frozen=True)
class RepairResult:
    final_outcome: VerificationOutcome
    rounds: tuple[dict[str, Any], ...]
    stopped_reason: str
    model_calls: int


@dataclass(frozen=True)
class PreparedRepairRequest:
    mode: Literal["SCHEMA", "SEMANTIC"]
    system_prompt: str
    user_prompt: str
    response_format: dict[str, Any]
    input_diagnostic: list[dict[str, Any]]
    input_warnings: list[dict[str, Any]]
    error_location: list[dict[str, Any]]
    grounding_output_domain: dict[str, Any]


def prepare_repair_request(
    *,
    instruction: str,
    outcome: VerificationOutcome,
    asg: ASG,
    previous_repair_diff: dict[str, Any] | None = None,
    previous_diagnostic: list[dict[str, Any]] | None = None,
    include_warnings: bool = False,
) -> PreparedRepairRequest:
    """Build one localized request without invoking a provider."""
    schema_mode = any(
        item.code == "SCHEMA_INVALID" and item.severity == "BLOCK"
        for item in outcome.diagnostics_after_repair
    )
    input_warnings = (
        build_warnings(outcome.diagnostics_after_repair) if include_warnings else []
    )
    if schema_mode:
        source = _mapping(outcome.prediction_artifact.get("semantic_source")) or {
            "raw_response": None,
            "candidate": outcome.prediction_artifact.get("semantic_contract"),
            "parse_error": None,
            "infrastructure_error": None,
        }
        input_diagnostic = build_error(outcome.diagnostics_after_repair)
        location = [
            {
                "handle": None,
                "semantic_type": "RAW_SEMANTIC_ARTIFACT",
                "role": "schema_invalid",
                "path": "$",
                "source_text": None,
            }
        ]
        output_label_domain = complete_asg_label_vocabulary(asg)
        label_grounding = _schema_grounding_view(outcome.prediction_artifact)
        system_prompt, user_prompt = build_schema_repair_request(
            instruction=instruction,
            semantic_source=source,
            schema_validation_error=input_diagnostic,
            label_grounding=label_grounding,
            grounding_output_label_domain=output_label_domain,
            error=input_diagnostic,
            error_location=location,
            previous_repair_diff=previous_repair_diff,
            previous_diagnostic=previous_diagnostic,
            deterministic_warnings=input_warnings,
        )
        return PreparedRepairRequest(
            mode="SCHEMA",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=repair_response_format(),
            input_diagnostic=input_diagnostic,
            input_warnings=input_warnings,
            error_location=location,
            grounding_output_domain={
                "passed": True,
                "mode": "SCHEMA",
                "visible_label_vocabulary": output_label_domain,
                "slots": [],
                "failures": [],
            },
        )

    contract = SceneIndependentSemanticContract.model_validate(
        outcome.prediction_artifact["semantic_contract"]
    )
    index = outcome.contract_index or build_contract_index(instruction, contract)
    input_diagnostic = build_error(outcome.diagnostics_after_repair)
    location = build_error_location(
        outcome.diagnostics_after_repair, index, contract
    )
    label_grounding = _label_grounding_view(
        contract, outcome.prediction_artifact, asg
    )
    output_label_domain = complete_asg_label_vocabulary(asg)
    output_domain_audit = build_grounding_output_domain_audit(
        contract,
        _mapping(outcome.prediction_artifact.get("label_grounding")),
        asg,
        index,
        visible_vocabulary=output_label_domain,
    )
    system_prompt, user_prompt = build_repair_request(
        instruction=instruction,
        semantic_contract=contract.model_dump(mode="json"),
        label_grounding=label_grounding,
        grounding_output_label_domain=output_label_domain,
        error=input_diagnostic,
        error_location=location,
        previous_repair_diff=previous_repair_diff,
        previous_diagnostic=previous_diagnostic,
        deterministic_warnings=input_warnings,
    )
    return PreparedRepairRequest(
        mode="SEMANTIC",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_format=repair_response_format(),
        input_diagnostic=input_diagnostic,
        input_warnings=input_warnings,
        error_location=location,
        grounding_output_domain=output_domain_audit,
    )


def run_semantic_repair(
    *,
    instruction: str,
    robot_position: Mapping[str, Any],
    starting_outcome: VerificationOutcome,
    asg: ASG,
    call_model: RepairCall,
    preflight: RepairPreflight | None = None,
    max_rounds: int = 2,
    include_warnings: bool = False,
    binding_resolver: Callable | None = None,
    execution_backend: Callable | None = None,
) -> RepairResult:
    """Repair only eligible semantic BLOCKs; never invoke other learned stages."""
    if max_rounds not in {0, 1, 2}:
        raise ValueError("repair permits at most two explicitly budgeted rounds")
    current = starting_outcome
    history = {_artifact_hash(current.prediction_artifact)}
    rounds: list[dict[str, Any]] = []
    model_calls = 0
    previous_diff: dict[str, Any] | None = None
    previous_diagnostic: list[dict[str, Any]] | None = None
    for round_number in range(1, max_rounds + 1):
        eligible, reason = repair_eligibility(
            final_status=current.final_status,
            diagnostics=current.diagnostics_after_repair,
        )
        if not eligible:
            return RepairResult(current, tuple(rounds), reason, model_calls)
        prepared = prepare_repair_request(
            instruction=instruction,
            outcome=current,
            asg=asg,
            previous_repair_diff=previous_diff,
            previous_diagnostic=previous_diagnostic,
            include_warnings=include_warnings,
        )
        old_contract = (
            SceneIndependentSemanticContract.model_validate(
                current.prediction_artifact["semantic_contract"]
            )
            if prepared.mode == "SEMANTIC"
            else None
        )
        system_prompt = prepared.system_prompt
        user_prompt = prepared.user_prompt
        response_format = prepared.response_format
        request_hash = _json_hash(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_format": response_format,
            }
        )
        if not prepared.grounding_output_domain["passed"]:
            rounds.append(
                {
                    "round": round_number,
                    "repair_mode": prepared.mode,
                    "request_hash": request_hash,
                    "response_hash": None,
                    "input_diagnostic": prepared.input_diagnostic,
                    "input_warnings": prepared.input_warnings,
                    "previous_diagnostic_supplied": previous_diagnostic,
                    "previous_repair_diff_supplied": previous_diff,
                    "error_location": prepared.error_location,
                    "request_stats": _request_stats(prepared),
                    "grounding_output_domain": prepared.grounding_output_domain,
                    "provider_called": False,
                    "failure_type": "incomplete_grounding_output_domain",
                    "validation": {
                        "success": False,
                        "errors": ["INCOMPLETE_GROUNDING_OUTPUT_DOMAIN"],
                    },
                }
            )
            return RepairResult(
                current,
                tuple(rounds),
                "incomplete grounding output domain",
                model_calls,
            )
        preflight_record = (
            preflight(system_prompt, user_prompt, response_format)
            if preflight is not None
            else {
                "input_characters": len(system_prompt) + len(user_prompt),
                "fits": True,
                "tokenizer": "not_configured",
            }
        )
        if not preflight_record.get("fits"):
            rounds.append(
                {
                    "round": round_number,
                    "repair_mode": prepared.mode,
                    "request_hash": request_hash,
                    "response_hash": None,
                    "input_diagnostic": prepared.input_diagnostic,
                    "input_warnings": prepared.input_warnings,
                    "previous_diagnostic_supplied": previous_diagnostic,
                    "previous_repair_diff_supplied": previous_diff,
                    "error_location": prepared.error_location,
                    "request_stats": _request_stats(prepared),
                    "grounding_output_domain": prepared.grounding_output_domain,
                    "context_preflight": preflight_record,
                    "provider_called": False,
                    "failure_type": "context_preflight_failure",
                    "validation": {
                        "success": False,
                        "errors": ["CONTEXT_PREFLIGHT_FAILURE"],
                    },
                }
            )
            return RepairResult(
                current, tuple(rounds), "context preflight failure", model_calls
            )
        started = time.perf_counter()
        model_calls += 1
        try:
            raw, call_record = call_model(system_prompt, user_prompt, response_format)
        except Exception as exc:  # noqa: BLE001 - provider boundary is recorded
            failure_type = _provider_failure_type(exc)
            rounds.append(
                {
                    "round": round_number,
                    "repair_mode": prepared.mode,
                    "request_hash": request_hash,
                    "response_hash": None,
                    "provider_error": f"{type(exc).__name__}: {exc}",
                    "provider_called": True,
                    "failure_type": failure_type,
                    "input_diagnostic": prepared.input_diagnostic,
                    "input_warnings": prepared.input_warnings,
                    "previous_diagnostic_supplied": previous_diagnostic,
                    "previous_repair_diff_supplied": previous_diff,
                    "error_location": prepared.error_location,
                    "request_stats": _request_stats(prepared),
                    "grounding_output_domain": prepared.grounding_output_domain,
                    "context_preflight": preflight_record,
                    "latency_seconds": time.perf_counter() - started,
                    "validation": {"success": False, "errors": ["provider failure"]},
                }
            )
            return RepairResult(current, tuple(rounds), failure_type, model_calls)
        response_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        parsed, errors = _parse_and_validate_output(
            raw, old_contract, current.prediction_artifact, asg
        )
        base_audit = {
            "round": round_number,
            "repair_mode": prepared.mode,
            "request_hash": request_hash,
            "response_hash": response_hash,
            "input_diagnostic": prepared.input_diagnostic,
            "input_warnings": prepared.input_warnings,
            "previous_diagnostic_supplied": previous_diagnostic,
            "previous_repair_diff_supplied": previous_diff,
            "error_location": prepared.error_location,
            "request_stats": _request_stats(prepared),
            "grounding_output_domain": prepared.grounding_output_domain,
            "context_preflight": preflight_record,
            "provider_called": True,
            "failure_type": None,
            # Full prompts/raw output already live in the shared model_calls
            # ledger. Keep only the reference here to avoid duplicating large
            # strings inside every assurance artifact.
            "call_record": call_record,
            "latency_seconds": time.perf_counter() - started,
            "validation": {"success": parsed is not None, "errors": errors},
        }
        if parsed is None:
            base_audit["failure_type"] = "invalid_semantic_output"
            rounds.append(base_audit)
            return RepairResult(
                current, tuple(rounds), "invalid repair output", model_calls
            )
        new_contract, new_label_grounding = parsed
        new_artifact = _repaired_runtime_artifact(
            current.prediction_artifact, new_contract, new_label_grounding
        )
        new_hash = _artifact_hash(new_artifact)
        semantic_diff = (
            _semantic_diff(old_contract, new_contract)
            if old_contract is not None
            else _json_diff(
                (_mapping(current.prediction_artifact.get("semantic_source")) or {}).get(
                    "candidate", current.prediction_artifact.get("semantic_contract")
                ),
                new_contract.model_dump(mode="json"),
            )
        )
        grounding_diff = _grounding_diff(
            current.prediction_artifact.get("label_grounding"), new_label_grounding
        )
        base_audit.update(
            {
                "old_contract_hash": (
                    _json_hash(old_contract.model_dump(mode="json"))
                    if old_contract is not None
                    else None
                ),
                "new_contract_hash": _json_hash(new_contract.model_dump(mode="json")),
                "semantic_diff": semantic_diff,
                "grounding_label_diff": grounding_diff,
                "determined_fact_validation": "PASS",
                "artifact_hash": new_hash,
            }
        )
        if not semantic_diff["changed"] and not grounding_diff["changed"]:
            base_audit["stop"] = "identical or repeated artifact hash"
            base_audit["stop_reason_code"] = "NO_CHANGE"
            base_audit["failure_type"] = "no_op_or_repeated_output"
            rounds.append(base_audit)
            return RepairResult(
                current, tuple(rounds), base_audit["stop"], model_calls
            )
        if new_hash in history:
            base_audit["stop"] = "identical or repeated artifact hash"
            base_audit["stop_reason_code"] = "REPEATED_ARTIFACT"
            base_audit["failure_type"] = "no_op_or_repeated_output"
            rounds.append(base_audit)
            return RepairResult(
                current, tuple(rounds), base_audit["stop"], model_calls
            )
        history.add(new_hash)
        refreshed = run_verification(
            instruction=instruction,
            robot_position=robot_position,
            artifact=new_artifact,
            asg=asg,
            force_recompute=True,
            binding_resolver=binding_resolver,
            execution_backend=execution_backend,
        )
        base_audit["verification_after"] = {
            "final_status": refreshed.final_status,
            "verification_accepted": refreshed.verification_accepted,
            "needs_backend_fix": refreshed.needs_backend_fix,
            "diagnostics": [
                item.model_dump(mode="json")
                for item in refreshed.diagnostics_after_repair
            ],
        }
        base_audit["post_forward_diagnostic"] = build_error(
            refreshed.diagnostics_after_repair
        )
        rounds.append(base_audit)
        current = refreshed
        if current.verification_accepted:
            return RepairResult(
                current, tuple(rounds), "accepted after repair", model_calls
            )
        if current.needs_backend_fix:
            return RepairResult(
                current,
                tuple(rounds),
                "failure became backend/infrastructure",
                model_calls,
            )
        previous_diff = {
            "semantic_diff": semantic_diff,
            "grounding_label_diff": grounding_diff,
        }
        previous_diagnostic = prepared.input_diagnostic
    return RepairResult(
        current, tuple(rounds), f"maximum {max_rounds} rounds used", model_calls
    )


def _request_stats(prepared: PreparedRepairRequest) -> dict[str, int]:
    return {"location_handles": len(prepared.error_location)}


def _provider_failure_type(exc: Exception) -> str:
    message = str(exc).lower()
    if "context length" in message or (
        "http 400" in message and "input_tokens" in message
    ):
        return "provider_context_rejection"
    return "provider_transport_failure"


def _parse_and_validate_output(
    raw: str,
    old_contract: SceneIndependentSemanticContract | None,
    old_artifact: Mapping[str, Any],
    asg: ASG,
) -> tuple[tuple[SceneIndependentSemanticContract, dict[str, Any]] | None, list[str]]:
    try:
        candidate = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, [f"JSON_PARSE_FAILED: {exc}"]
    forbidden = sorted(_forbidden_keys(candidate))
    if forbidden:
        return None, [f"NUMERIC_SCENE_OUTPUT_FORBIDDEN: {forbidden}"]
    try:
        output = RepairOutput.model_validate(candidate)
    except ValidationError as exc:
        return None, [f"SCHEMA_VALIDATION_FAILED: {exc}"]
    new_contract = output.semantic_contract
    collection = collect_nominal_commitments(new_contract)
    bindings = output.grounding_bindings
    errors: list[str] = []
    if len(bindings) != len(collection.commitments):
        errors.append(
            f"CARDINALITY_MISMATCH: expected {len(collection.commitments)}, got {len(bindings)}"
        )
    vocabulary = vocabulary_from_entity_payload(asg.to_entity_dict())
    determine = determine_groundings(collection, vocabulary)
    structural = dict(
        zip(
            determine.value("grounding.structurally_determined_positions", ()),
            determine.value("grounding.structurally_determined", ()),
        )
    )
    fully = dict(
        zip(
            determine.value("grounding.fully_determined_positions", ()),
            determine.value("grounding.fully_determined", ()),
        )
    )
    requested = []
    for position, commitment in enumerate(collection.commitments):
        requested.append(
            structural.get(position)
            or _grounding_item(commitment.head_text, commitment.source_text)
        )
    errors.extend(validate_bindings(requested, bindings, vocabulary))
    for position, binding in enumerate(bindings):
        if position in fully and binding != fully[position]:
            errors.append(f"DETERMINED_BINDING_CHANGED[index={position}]")
    if old_contract is not None:
        errors.extend(
            _unchanged_fact_errors(
                old_contract, old_artifact, new_contract, bindings, asg
            )
        )
    if errors:
        return None, list(dict.fromkeys(errors))
    return (new_contract, _label_grounding_artifact(new_contract, bindings, determine)), []


def _unchanged_fact_errors(
    old_contract: SceneIndependentSemanticContract,
    old_artifact: Mapping[str, Any],
    new_contract: SceneIndependentSemanticContract,
    bindings: list[GroundingBinding],
    asg: ASG,
) -> list[str]:
    vocabulary = vocabulary_from_entity_payload(asg.to_entity_dict())
    old_collection = collect_nominal_commitments(old_contract)
    old_determine = determine_groundings(old_collection, vocabulary)
    new_collection = collect_nominal_commitments(new_contract)
    old_by_path = {item.path: (position, item) for position, item in enumerate(old_collection.commitments)}
    old_full = dict(
        zip(
            old_determine.value("grounding.fully_determined_positions", ()),
            old_determine.value("grounding.fully_determined", ()),
        )
    )
    old_structural = dict(
        zip(
            old_determine.value("grounding.structurally_determined_positions", ()),
            old_determine.value("grounding.structurally_determined", ()),
        )
    )
    errors = []
    for new_position, item in enumerate(new_collection.commitments):
        prior = old_by_path.get(item.path)
        if prior is None or new_position >= len(bindings):
            continue
        old_position, old_item = prior
        if (
            item.head_text,
            item.source_text,
            item.possible_kinds,
            item.structural_reasons,
        ) != (
            old_item.head_text,
            old_item.source_text,
            old_item.possible_kinds,
            old_item.structural_reasons,
        ):
            continue
        if old_position in old_full and bindings[new_position] != old_full[old_position]:
            errors.append(f"UNCHANGED_DETERMINED_BINDING_OVERWRITTEN[path={item.path}]")
        elif (
            old_position in old_structural
            and bindings[new_position].asg_kind != old_structural[old_position].asg_kind
        ):
            errors.append(f"UNCHANGED_DETERMINED_KIND_OVERWRITTEN[path={item.path}]")
    return errors


def _label_grounding_view(
    contract: SceneIndependentSemanticContract,
    artifact: Mapping[str, Any],
    asg: ASG,
) -> list[dict[str, Any]]:
    collection = collect_nominal_commitments(contract)
    determine = determine_groundings(
        collection, vocabulary_from_entity_payload(asg.to_entity_dict())
    )
    full = dict(
        zip(
            determine.value("grounding.fully_determined_positions", ()),
            determine.value("grounding.fully_determined", ()),
        )
    )
    structural = dict(
        zip(
            determine.value("grounding.structurally_determined_positions", ()),
            determine.value("grounding.structurally_determined", ()),
        )
    )
    raw = _mapping(artifact.get("label_grounding")) or {}
    bindings = raw.get("final_bindings") if isinstance(raw.get("final_bindings"), list) else []
    return [
        {
            "position": position,
            "path": item.path,
            "source_text": item.source_text,
            "head_text": item.head_text,
            "possible_kinds": sorted(item.possible_kinds),
            "current_binding": bindings[position] if position < len(bindings) else None,
            "determination": "fully_determined" if position in full else "kind_determined" if position in structural else "residual",
            "determined_binding": full[position].model_dump(mode="json") if position in full else None,
            "determined_kind": structural[position].asg_kind if position in structural else None,
        }
        for position, item in enumerate(collection.commitments)
    ]


def _schema_grounding_view(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = _mapping(artifact.get("label_grounding")) or {}
    items = raw.get("all_grounding_items")
    bindings = raw.get("final_bindings")
    safe_items = items if isinstance(items, list) else []
    safe_bindings = bindings if isinstance(bindings, list) else []
    return [
        {
            "position": position,
            "grounding_item": safe_items[position] if position < len(safe_items) else None,
            "current_binding": binding,
        }
        for position, binding in enumerate(safe_bindings)
    ]


def _label_grounding_artifact(
    contract: SceneIndependentSemanticContract,
    bindings: list[GroundingBinding],
    determine: Any,
) -> dict[str, Any]:
    collection = collect_nominal_commitments(contract)
    structural = dict(
        zip(
            determine.value("grounding.structurally_determined_positions", ()),
            determine.value("grounding.structurally_determined", ()),
        )
    )
    return {
        "stage1_gate_passed": True,
        "occurrence_count": len(collection.commitments),
        "all_grounding_items": [
            {
                "head_text": item.head_text,
                "source_text": item.source_text,
                "asg_kind": structural[position].asg_kind if position in structural else None,
            }
            for position, item in enumerate(collection.commitments)
        ],
        "final_bindings": [item.model_dump(mode="json") for item in bindings],
        "validation": {"executed": True, "parse_success": True, "success": True, "diagnostics": []},
        "llm_call_accounting": {
            "logical_calls": 0,
            "usage": {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "llm_latency_seconds": 0.0},
            "attempt_count": 0,
            "attempts": [],
            "error": None,
            "semantic_retry_count": 0,
        },
    }


def _repaired_runtime_artifact(
    artifact: Mapping[str, Any],
    contract: SceneIndependentSemanticContract,
    label_grounding: dict[str, Any],
) -> dict[str, Any]:
    value = copy.deepcopy(dict(artifact))
    for key in _STALE_KEYS:
        value.pop(key, None)
    value["semantic_contract"] = contract.model_dump(mode="json")
    value["label_grounding"] = label_grounding
    return value


def _semantic_diff(
    old: SceneIndependentSemanticContract, new: SceneIndependentSemanticContract
) -> dict[str, Any]:
    left = old.model_dump(mode="json")
    right = new.model_dump(mode="json")
    return _json_diff(left, right)


def _grounding_diff(old: Any, new: Mapping[str, Any]) -> dict[str, Any]:
    old_bindings = _mapping(old).get("final_bindings", []) if _mapping(old) else []
    return _json_diff(old_bindings, new.get("final_bindings", []))


def _json_diff(left: Any, right: Any) -> dict[str, Any]:
    if left == right:
        return {"changed": False, "paths": [], "edit_count": 0}
    paths: list[str] = []

    def visit(a: Any, b: Any, path: str) -> None:
        if a == b:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                visit(a.get(key), b.get(key), f"{path}.{key}")
            return
        if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f"{path}[{index}]")
            return
        paths.append(path)

    visit(left, right, "$")
    return {"changed": True, "paths": paths, "edit_count": len(paths)}


def _forbidden_keys(value: Any) -> set[str]:
    hits: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_OUTPUT_KEYS:
                hits.add(str(key))
            hits.update(_forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            hits.update(_forbidden_keys(child))
    return hits


def _artifact_hash(value: Mapping[str, Any]) -> str:
    label_grounding = _mapping(value.get("label_grounding")) or {}
    return _json_hash(
        {
            "semantic_contract": value.get("semantic_contract"),
            "grounding_bindings": label_grounding.get("final_bindings", []),
        }
    )


def _json_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _mapping(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _grounding_item(head_text: str, source_text: str) -> Any:
    from reconav.method.grounding.contracts import GroundingItem

    return GroundingItem(head_text=head_text, source_text=source_text, asg_kind=None)
