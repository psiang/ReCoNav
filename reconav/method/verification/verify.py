"""Deterministic verification assurance and commitment-preserving repair."""

from __future__ import annotations

import copy
import re
import time
from collections.abc import Mapping, Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from reconav.benchmark.metrics import replayable_route
from reconav.benchmark.route_evaluator import evaluate_grounded_route
from reconav.scene.asg import ASG
from reconav.solver.avoid_checker import AvoidConsistencyError
from reconav.solver.cancellation import SolveCancelledError
from reconav.solver.count_checker import CountConsistencyError
from reconav.solver.execution_profiles import ExecutionProfile
from reconav.solver.order_checker import OrderConsistencyError
from reconav.solver.backend import solve_dsl_route
from reconav.method.verification.contract_index import ContractIndex, build_contract_index
from reconav.method.verification.diagnostics import (
    VerificationDiagnostic,
    VerificationStatus,
    status_for,
)
from reconav.method.verification.semantic_contract import (
    NONBLOCKING_DIAGNOSTIC_CODES,
    audit_semantic_contract,
)
from reconav.method.determination import (
    ActionEffect,
    CommitmentLedger,
    Derivation,
    Determination,
    DeterminedFact,
    DeterministicAction,
    Diagnostic,
    RuleRegistry,
)
from reconav.method.grounding.contracts import ASGVocabulary, GroundingBinding
from reconav.method.grounding.grounded_contract import SceneGroundedContract
from reconav.method.grounding.label_grounding import (
    CommitmentCollection,
    collect_nominal_commitments,
    determine_groundings,
    validate_bindings,
    vocabulary_from_entity_payload,
)
from reconav.method.grounding.scene_binding import bind_scene, binding_for_referent
from reconav.method.lowering.dsl import lower_grounded_contract
from reconav.method.semantic.contracts import SceneIndependentSemanticContract

_INVALIDATED_AFTER_GROUNDING = [
    "ASG_INSTANCE_BINDING",
    "GROUNDED_CONTRACT",
    "DSL_LOWERING",
    "SOLVER_RESULT",
    "ROUTE",
    "WAYPOINTS",
    "EVALUATOR_PREDICTION",
]
_PROVENANCE_WARNING_CODES = NONBLOCKING_DIAGNOSTIC_CODES
_RUNTIME_ARTIFACT_KEYS = (
    "semantic_contract",
    "semantic_source",
    "label_grounding",
    "scene_binding",
    "lowering",
    "execution",
    "predicted",
    "route",
    "stage_verification",
    "failure",
    "timing",
    "candidate_resolution",
)
_EMPTY_OPTIONAL_COLLECTION_KEYS = frozenset(
    {
        "qualifiers",
        "via",
        "order",
        "avoid",
        "count",
        "carry",
        "width",
        "prefer",
        "prefer_order",
    }
)
_DETERMINISTIC_CONSISTENCY_ERRORS = (
    OrderConsistencyError,
    AvoidConsistencyError,
    CountConsistencyError,
)


def _registry_assertion(rule_id: str, fact_key: str):
    """Expose the language checks through the strict shared registry contract."""
    def check(ledger: CommitmentLedger):
        for fact in ledger.facts:
            if fact.key != fact_key:
                continue
            valid = fact.value is not None and fact.value is not False
            yield Diagnostic(
                code=f"{fact_key.upper().replace('.', '_')}_{'VALID' if valid else 'INVALID'}",
                severity="PASS" if valid else "BLOCK",
                subject=fact.subject,
                message="language-stage contract evidence is present" if valid else "language-stage contract evidence is missing",
                assertion=rule_id,
                supporting_facts=[fact.fact_id or ""],
                responsible_commitment=fact.responsible_commitment,
                entity_kind=fact.entity_kind,
                entity_id=fact.entity_id,
            )
    return check


def _registry_evidence(rule_id: str, fact_key: str):
    def collect(ledger: CommitmentLedger):
        for fact in ledger.facts:
            if fact.key == fact_key:
                yield DeterminedFact(
                    key=f"{fact_key}.evidence", subject=fact.subject, value=fact.value,
                    origin="SOLVER", rule=rule_id,
                    derivation=Derivation(rule=rule_id, premises=[fact.fact_id or ""]),
                    responsible_commitment=fact.responsible_commitment,
                    entity_kind=fact.entity_kind, entity_id=fact.entity_id,
                )
    return collect


def _registry_repair(rule_id: str, fact_key: str):
    """Materialize only an explicitly determined language repair request."""
    def repair(ledger: CommitmentLedger):
        for fact in ledger.facts:
            if fact.key != fact_key or not isinstance(fact.value, Mapping):
                continue
            request = fact.value.get("deterministic_repair")
            if not isinstance(request, Mapping):
                continue
            effect = request.get("effect")
            if effect not in {"CANONICALIZE", "RESTORE", "RESOLVE"}:
                continue
            yield DeterministicAction(
                checkpoint=request.get("checkpoint", "label_grounding"), effect=effect,
                target=str(request.get("target", fact.subject)), before=request.get("before"),
                after=request.get("after"), rule=rule_id, supporting_facts=[fact.fact_id or ""],
                responsible_commitment=fact.responsible_commitment,
                entity_kind=fact.entity_kind, entity_id=fact.entity_id,
            )
    return repair


def language_rule_registry() -> RuleRegistry:
    registry = RuleRegistry()
    registry.register(
        "language.semantic.contract_schema",
        capabilities={"ASSERT", "EVIDENCE"},
        assert_fn=_registry_assertion("language.semantic.contract_schema", "semantic.contract"),
        evidence_fn=_registry_evidence("language.semantic.contract_schema", "semantic.contract"),
    )
    registry.register(
        "grounding.commitment.binding",
        capabilities={"ASSERT", "REPAIR", "EVIDENCE"},
        assert_fn=_registry_assertion("grounding.commitment.binding", "grounding.binding"),
        repair_fn=_registry_repair("grounding.commitment.binding", "grounding.binding"),
        evidence_fn=_registry_evidence("grounding.commitment.binding", "grounding.binding"),
    )
    registry.register(
        "language.lowering.commitment_preservation",
        capabilities={"ASSERT", "REPAIR", "EVIDENCE"},
        assert_fn=_registry_assertion("language.lowering.commitment_preservation", "lowering.commitment_preservation"),
        repair_fn=_registry_repair("language.lowering.commitment_preservation", "lowering.commitment_preservation"),
        evidence_fn=_registry_evidence("language.lowering.commitment_preservation", "lowering.commitment_preservation"),
    )
    registry.register(
        "planning.route.contract_satisfaction",
        capabilities={"ASSERT", "EVIDENCE"},
        assert_fn=_registry_assertion("planning.route.contract_satisfaction", "route.contract_satisfaction"),
        evidence_fn=_registry_evidence("planning.route.contract_satisfaction", "route.contract_satisfaction"),
    )
    return registry


LANGUAGE_RULE_REGISTRY = language_rule_registry()


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class VerificationOutcome(_StrictModel):
    contract_index: ContractIndex | None
    determination: Determination
    diagnostics_before_repair: list[VerificationDiagnostic]
    diagnostics_after_repair: list[VerificationDiagnostic]
    initial_status: VerificationStatus
    final_status: VerificationStatus
    verification_accepted: bool
    backend_assertion_pass: bool | None
    needs_backend_fix: bool
    repair_iterations: int
    verification_llm_calls: int
    additional_forward_llm_calls: int
    label_grounding_reexecution_count: int
    new_label_grounding_residual_llm_calls: int
    invalidated_descendants: list[str]
    prediction_artifact: dict[str, Any]


def run_verification(
    *,
    instruction: str,
    robot_position: Mapping[str, Any],
    artifact: Mapping[str, Any],
    asg: ASG,
    enabled_repairs: frozenset[ActionEffect] = frozenset(
        {"CANONICALIZE", "RESTORE", "RESOLVE"}
    ),
    force_recompute: bool = False,
    binding_resolver: Callable | None = None,
    execution_backend: Callable | None = None,
) -> VerificationOutcome:
    """Assure one frozen prediction without model labels, hidden gold, or LLM calls."""
    working = copy.deepcopy(dict(artifact))
    raw_contract = working.get("semantic_contract")
    try:
        contract = SceneIndependentSemanticContract.model_validate(raw_contract)
    except ValidationError as exc:
        semantic_source = working.get("semantic_source")
        source = semantic_source if isinstance(semantic_source, Mapping) else {}
        content_returned = (
            raw_contract is not None
            or source.get("candidate") is not None
            or bool(source.get("raw_response"))
        )
        infrastructure_only = bool(source.get("infrastructure_error")) and not content_returned
        diagnostic = _diagnostic(
            code=(
                "UPSTREAM_INFRASTRUCTURE_FAILURE"
                if infrastructure_only
                else "SCHEMA_INVALID"
            ),
            checkpoint="contract_validation",
            category="ARTIFACT",
            severity="BLOCK",
            node_or_path="$",
            assertion=(
                "upstream provider returns semantic content"
                if infrastructure_only
                else "canonical Contract passes strict schema validation"
            ),
            expected=(
                "semantic content available for validation"
                if infrastructure_only
                else "SceneIndependentSemanticContract"
            ),
            observed=(source.get("infrastructure_error") if infrastructure_only else str(exc)),
            detail={
                "schema_validation_error": str(exc),
                "semantic_source_available": bool(source),
                "content_returned": content_returned,
            },
        )
        return _blocked_outcome(working, [diagnostic])

    index = build_contract_index(instruction, contract)
    s1 = _stage1_diagnostics(instruction, contract, index)
    collection = collect_nominal_commitments(contract)
    vocabulary = vocabulary_from_entity_payload(asg.to_entity_dict())
    determination = determine_groundings(collection, vocabulary)
    determination = determination.extended(
        facts=[
            *_contract_index_facts(index),
            DeterminedFact(
                key="rule.registry",
                subject="language.verification",
                value=[rule.rule_id for rule in LANGUAGE_RULE_REGISTRY.rules()],
                origin="PROPAGATION",
                rule="language.rule_registry.active",
            ),
        ]
    )
    unrepairable_s1 = any(item.severity == "BLOCK" for item in s1)
    grounding_before, actions = _grounding_diagnostics(
        contract,
        working.get("label_grounding"),
        collection,
        determination,
        vocabulary,
        allow_repairs=frozenset() if unrepairable_s1 else enabled_repairs,
    )
    diagnostics_before = [*s1, *grounding_before]
    semantic_repair = bool(actions)
    if semantic_repair:
        working["label_grounding"] = _apply_grounding_actions(
            working.get("label_grounding"), collection, determination, actions
        )
        determination = determination.extended(actions=actions)
        invalidate_frozen = getattr(
            binding_resolver, "invalidate_frozen", None
        )
        if callable(invalidate_frozen):
            # The progressive frontier moved: candidate resolution feasibility/preference
            # evidence was proved for the pre-repair grounding and must not be
            # replayed after any deterministic verification grounding repair.
            invalidate_frozen()

    if unrepairable_s1:
        return _finish(
            working=working,
            index=index,
            determination=determination,
            before=diagnostics_before,
            after=s1,
            iterations=1 if semantic_repair else 0,
        )

    if semantic_repair:
        grounding_after, _unused_actions = _grounding_diagnostics(
            contract,
            working.get("label_grounding"),
            collection,
            determination,
            vocabulary,
            allow_repairs=frozenset(),
        )
    else:
        # No artifact changed, so the second deterministic grounding pass
        # would be byte-for-byte equivalent to the first one.
        grounding_after = grounding_before
    if any(item.severity == "BLOCK" for item in grounding_after):
        return _finish(
            working=working,
            index=index,
            determination=determination,
            before=diagnostics_before,
            after=[*s1, *grounding_after],
            iterations=1 if semantic_repair else 0,
        )

    label_grounding = _executable_label_grounding(working["label_grounding"], len(collection.commitments))
    scene, binding_diagnostics = _bind_and_assert(
        contract, label_grounding, asg, robot_position,
        binding_resolver=binding_resolver, working=working,
    )
    resolution = working.get("candidate_resolution")
    if scene is not None and isinstance(resolution, Mapping):
        mechanism = str(resolution.get("mechanism", "candidate_resolution"))
        determination = determination.extended(facts=[DeterminedFact(
            key="grounding.full_program_identity", subject="scene_binding",
            value={"mechanism": mechanism, "chosen": resolution.get("chosen"),
                   "complete": resolution.get("complete")},
            origin="PROPAGATION" if mechanism == "local_unique" else "SOLVER",
            rule="LOCAL_DENOTATION_UNIQUENESS" if mechanism == "local_unique" else
                 "EXPLICIT_PREFERENCE_SELECTION" if mechanism == "preference_selected" else "FULL_PROGRAM_CANDIDATE_FEASIBILITY",
        )])
    if scene is None:
        after = [*s1, *grounding_after, *binding_diagnostics]
        return _finish(
            working=working,
            index=index,
            determination=determination,
            before=diagnostics_before if semantic_repair else after,
            after=after,
            iterations=1 if semantic_repair else 0,
        )

    scene_stale = not _scene_artifact_matches(working, scene)
    _store_scene_artifact(working, contract, label_grounding, scene)
    if any(item.severity == "BLOCK" for item in binding_diagnostics):
        after = [*s1, *grounding_after, *binding_diagnostics]
        return _finish(
            working=working,
            index=index,
            determination=_with_runtime_facts(
                determination,
                contract=contract,
                index=index,
                scene=scene,
                lowering=None,
                working=working,
            ),
            before=diagnostics_before if semantic_repair else after,
            after=after,
            iterations=1 if semantic_repair else 0,
        )
    lowering, lowering_diagnostics = _lower_and_assert(
        contract, scene, robot_position, asg
    )
    if lowering is None:
        after = [*s1, *grounding_after, *binding_diagnostics, *lowering_diagnostics]
        return _finish(
            working=working,
            index=index,
            determination=_with_runtime_facts(
                determination,
                contract=contract,
                index=index,
                scene=scene,
                lowering=None,
                working=working,
            ),
            before=diagnostics_before if semantic_repair else after,
            after=after,
            iterations=1 if semantic_repair else 0,
        )
    if any(item.severity == "BLOCK" for item in lowering_diagnostics):
        after = [*s1, *grounding_after, *binding_diagnostics, *lowering_diagnostics]
        return _finish(
            working=working,
            index=index,
            determination=_with_runtime_facts(
                determination,
                contract=contract,
                index=index,
                scene=scene,
                lowering=None,
                working=working,
            ),
            before=diagnostics_before if semantic_repair else after,
            after=after,
            iterations=1 if semantic_repair else 0,
        )

    stale_parts = _stale_lowering_parts(working, lowering)
    recompute = semantic_repair or force_recompute or scene_stale or bool(stale_parts)
    historical_stale = (
        not semantic_repair
        and not force_recompute
        and (scene_stale or bool(stale_parts))
    )
    stale_diagnostic = None
    if historical_stale:
        stale_diagnostic = _diagnostic(
            code="STALE_DERIVED_ARTIFACT_RECOMPUTED",
            checkpoint="route_lowering",
            category="ARTIFACT",
            severity="WARNING",
            node_or_path="$.scene_binding" if scene_stale else "$.lowering",
            assertion="saved descendants match the current Contract and grounding",
            expected="current deterministic derivation",
            observed={
                "scene_binding_stale": scene_stale,
                "stale_parts": stale_parts,
            },
        )
        determination = determination.extended(
            facts=[
                DeterminedFact(
                    key="artifact.stale_recomputed",
                    subject=stale_diagnostic.node_or_path,
                    value=stale_diagnostic.observed,
                    origin="PROPAGATION",
                    rule="CURRENT_COMMITMENT_RECOMPUTATION",
                )
            ]
        )
    _store_lowering_artifact(working, lowering)

    execution_diagnostics = _execute_and_assert(
        working,
        contract,
        label_grounding,
        scene,
        lowering,
        robot_position,
        asg,
        recompute=recompute,
        execution_backend=execution_backend,
    )
    after = [
        *s1,
        *grounding_after,
        *binding_diagnostics,
        *lowering_diagnostics,
        *execution_diagnostics,
    ]
    if not semantic_repair:
        diagnostics_before = [*after, *([stale_diagnostic] if stale_diagnostic else [])]
    return _finish(
        working=working,
        index=index,
        determination=_with_runtime_facts(
            determination,
            contract=contract,
            index=index,
            scene=scene,
            lowering=lowering,
            working=working,
        ),
        before=diagnostics_before,
        after=after,
        iterations=1 if semantic_repair else 0,
        backend_assertion_pass=_backend_assertion_pass(
            execution_diagnostics, working
        ),
    )


def _stage1_diagnostics(
    instruction: str,
    contract: SceneIndependentSemanticContract,
    index: ContractIndex,
) -> list[VerificationDiagnostic]:
    diagnostics: list[VerificationDiagnostic] = []
    audit = audit_semantic_contract(instruction, contract)
    for item in audit.diagnostics:
        warning = item.code in _PROVENANCE_WARNING_CODES
        diagnostics.append(
            _diagnostic(
                code=item.code,
                checkpoint="contract_validation",
                category="ARTIFACT" if warning else "COMMITMENT",
                severity="WARNING" if warning else "BLOCK",
                node_or_path=item.path,
                assertion=(
                    "source evidence is canonical bookkeeping"
                    if warning
                    else "semantic identity and structure are deterministically executable"
                ),
                expected="canonical source provenance"
                if warning
                else "consistent semantic structure",
                observed=item.message,
                detail={"historical_determination": item.determination},
            )
        )
    handles = [item.handle for item in index.nodes]
    if len(handles) != len(set(handles)):
        diagnostics.append(
            _diagnostic(
                code="DUPLICATE_CONTRACT_HANDLE",
                checkpoint="contract_validation",
                category="ARTIFACT",
                severity="BLOCK",
                node_or_path="$",
                assertion="R/E/C handles are unique",
                expected=len(handles),
                observed=len(set(handles)),
            )
        )
    for edge in index.edges:
        if edge.kind != "EVENT_REFERENCE" or edge.resolution == "UNIQUE_TEXT":
            continue
        code = (
            "DANGLING_EVENT_REFERENCE"
            if edge.resolution == "DANGLING"
            else "AMBIGUOUS_EVENT_REFERENCE"
        )
        if not any(
            item.code
            in {
                code,
                "EVENT_REFERENCE_TARGET_NOT_DECLARED",
                "AMBIGUOUS_EVENT_REFERENCE",
            }
            and item.node_or_path == edge.path
            for item in diagnostics
        ):
            diagnostics.append(
                _diagnostic(
                    code=code,
                    checkpoint="contract_validation",
                    category="COMMITMENT",
                    severity="BLOCK",
                    node_or_path=edge.path,
                    assertion="event reference resolves to exactly one declared E handle",
                    expected=1,
                    observed=edge.resolution,
                    detail={"target_source_text": edge.target_source_text},
                )
            )
    return _deduplicate(diagnostics)


def _grounding_diagnostics(
    contract: SceneIndependentSemanticContract,
    raw_label_grounding: Any,
    collection: CommitmentCollection,
    determination: Determination,
    vocabulary: ASGVocabulary,
    *,
    allow_repairs: frozenset[ActionEffect],
) -> tuple[list[VerificationDiagnostic], list[DeterministicAction]]:
    del contract
    diagnostics: list[VerificationDiagnostic] = []
    actions: list[DeterministicAction] = []
    for item in collection.diagnostics:
        diagnostics.append(
            _diagnostic(
                code=item.code,
                checkpoint="label_grounding",
                category="COMMITMENT",
                severity="BLOCK",
                node_or_path=item.path,
                assertion="authoritative kind requirements have a non-empty intersection",
                expected="at least one ASG kind",
                observed=item.detail,
                affected_commitments=[item.path],
            )
        )
    if diagnostics:
        return diagnostics, actions

    label_grounding = copy.deepcopy(raw_label_grounding) if isinstance(raw_label_grounding, Mapping) else {}
    items = label_grounding.get("all_grounding_items")
    bindings = label_grounding.get("final_bindings")
    validation = label_grounding.get("validation")
    available = (
        isinstance(items, list)
        and isinstance(bindings, list)
        and isinstance(validation, Mapping)
        and validation.get("success") is True
        and len(items) == len(collection.commitments)
        and len(bindings) == len(collection.commitments)
    )
    fully_determined_positions = tuple(
        determination.value("grounding.fully_determined_positions", ())
    )
    fully_determined = tuple(
        determination.value("grounding.fully_determined", ())
    )
    residual_positions = tuple(
        determination.value("grounding.residual_positions", ())
    )
    determined = dict(zip(fully_determined_positions, fully_determined))
    if not available:
        fully_unique = len(determined) == len(collection.commitments)
        repairable = fully_unique
        diagnostics.append(
            _diagnostic(
                code="GROUNDING_ARTIFACT_MISSING",
                checkpoint="label_grounding",
                category="GROUNDING",
                severity="BLOCK",
                node_or_path="$.label_grounding",
                assertion="every nominal occurrence has a validated binding",
                expected=len(collection.commitments),
                observed=len(bindings) if isinstance(bindings, list) else 0,
                detail={
                    "unique_canonical_positions": sorted(determined),
                    "residual_positions": list(residual_positions),
                },
                supporting_facts=[
                    f"grounding.{position}.canonical_binding"
                    for position in sorted(determined)
                ],
                affected_commitments=[item.path for item in collection.commitments],
                repairable=repairable,
            )
        )
        if not fully_unique or "RESOLVE" not in allow_repairs:
            return diagnostics, actions
        actions.append(
            DeterministicAction(
                effect="RESOLVE",
                checkpoint="label_grounding",
                rule="UNIQUE_CANONICAL_GROUNDING_SET",
                target="$.label_grounding",
                before=label_grounding,
                after=_synthesized_label_grounding(collection, determination),
                supporting_facts=[
                    f"grounding.{position}.canonical_binding"
                    for position in sorted(determined)
                ],
                invalidates=_INVALIDATED_AFTER_GROUNDING,
                verified=True,
            )
        )
        return diagnostics, actions

    structural_positions = tuple(
        determination.value("grounding.structurally_determined_positions", ())
    )
    structural_items = tuple(
        determination.value("grounding.structurally_determined", ())
    )
    structural = dict(zip(structural_positions, structural_items))
    for position, commitment in enumerate(collection.commitments):
        raw = bindings[position]
        try:
            current = GroundingBinding.model_validate(raw)
        except ValidationError as exc:
            current = None
            parse_error = str(exc)
        else:
            parse_error = None
        canonical = determined.get(position)
        required = structural.get(position)
        problems: list[str] = []
        if current is None:
            problems.append("invalid binding schema")
        else:
            if current.head_text != commitment.head_text:
                problems.append("head_text differs from authoritative occurrence")
            if required is not None and current.asg_kind != required.asg_kind:
                problems.append("kind violates structural requirement")
            if validate_bindings(
                [required or _request_item(commitment)], [current], vocabulary
            ):
                problems.append("binding violates vocabulary or structural hint")
            if canonical is not None and current != canonical:
                problems.append("binding differs from unique canonical lexical match")
        if not problems:
            continue
        replacement = canonical
        effect: ActionEffect | None = (
            "RESTORE" if required is not None else "RESOLVE"
        )
        if replacement is None and current is not None and required is not None:
            labels = getattr(vocabulary, required.asg_kind)
            if current.asg_label in labels:
                replacement = current.model_copy(update={"asg_kind": required.asg_kind})
                effect = "RESTORE"
        repairable = replacement is not None
        diagnostics.append(
            _diagnostic(
                code="GROUNDING_COMMITMENT_CONFLICT",
                checkpoint="label_grounding",
                category="GROUNDING",
                severity="BLOCK",
                node_or_path=commitment.path,
                assertion="binding agrees with structural and canonical deterministic facts",
                expected=replacement.model_dump(mode="json") if replacement else None,
                observed=raw if parse_error is None else parse_error,
                detail={
                    "reasons": problems,
                    "structural_reasons": list(commitment.structural_reasons),
                    "canonical_match_unique": canonical is not None,
                },
                supporting_facts=[
                    key
                    for key in (
                        f"grounding.{position}.structural_binding"
                        if required is not None
                        else None,
                        f"grounding.{position}.canonical_binding"
                        if canonical is not None
                        else None,
                    )
                    if key is not None
                ],
                affected_commitments=[commitment.path],
                repairable=repairable,
            )
        )
        if not repairable or effect not in allow_repairs:
            continue
        supporting_facts = [
            key
            for key in (
                f"grounding.{position}.structural_binding"
                if required is not None
                else None,
                f"grounding.{position}.canonical_binding"
                if canonical is not None
                else None,
            )
            if key is not None
        ]
        actions.append(
            DeterministicAction(
                effect=effect,
                checkpoint="label_grounding",
                rule="RESTORE_AUTHORITATIVE_GROUNDING"
                if effect == "RESTORE"
                else "UNIQUE_CANONICAL_LABEL",
                target=commitment.path,
                before=raw,
                after=replacement.model_dump(mode="json"),
                supporting_facts=supporting_facts,
                invalidates=_INVALIDATED_AFTER_GROUNDING,
                verified=True,
            )
        )
    return diagnostics, actions


def _apply_grounding_actions(
    raw_label_grounding: Any,
    collection: CommitmentCollection,
    determination: Determination,
    actions: list[DeterministicAction],
) -> dict[str, Any]:
    label_grounding = copy.deepcopy(raw_label_grounding) if isinstance(raw_label_grounding, Mapping) else {}
    for action in actions:
        if action.target == "$.label_grounding":
            label_grounding = copy.deepcopy(action.after)
            continue
        position = next(
            index
            for index, item in enumerate(collection.commitments)
            if item.path == action.target
        )
        commitment = collection.commitments[position]
        structural = determination.value(
            f"grounding.{position}.structural_binding"
        )
        label_grounding["final_bindings"][position] = copy.deepcopy(action.after)
        label_grounding["all_grounding_items"][position] = {
            "head_text": commitment.head_text,
            "source_text": commitment.source_text,
            "asg_kind": structural.asg_kind if structural is not None else None,
        }
    label_grounding["validation"] = {
        "executed": True,
        "parse_success": True,
        "success": True,
        "diagnostics": [],
    }
    label_grounding["occurrence_count"] = len(collection.commitments)
    label_grounding["stage1_gate_passed"] = True
    return label_grounding


def _bind_and_assert(
    contract: SceneIndependentSemanticContract,
    label_grounding: Mapping[str, Any],
    asg: ASG,
    robot_position: Mapping[str, Any],
    *, binding_resolver: Callable | None = None, working: dict | None = None,
) -> tuple[SceneGroundedContract | None, list[VerificationDiagnostic]]:
    try:
        if binding_resolver is None:
            scene = bind_scene(contract, label_grounding, asg, robot_position)
        else:
            scene, resolution = binding_resolver(contract, label_grounding, asg, robot_position)
            if working is not None:
                working["candidate_resolution"] = resolution
    except Exception as exc:  # noqa: BLE001 - checkpoint records backend failures
        message = str(exc)
        if working is not None and hasattr(exc, "report"):
            working["candidate_resolution"] = exc.report
        code = getattr(exc, "code", None) or (
            "IDENTITY_REQUIRED_EMPTY"
            if "candidates []" in message or "no candidates" in message
            else "IDENTITY_REQUIRED_AMBIGUOUS"
        )
        return None, [
            _diagnostic(
                code=code,
                checkpoint="scene_binding",
                category="GROUNDING",
                severity="BLOCK",
                node_or_path="$.scene_binding",
                assertion="required identities are locally unique or jointly resolved by the complete hard theory / explicit preference",
                expected="executable candidate cardinality",
                observed=message,
                detail={"exception_type": type(exc).__name__},
            )
        ]
    diagnostics: list[VerificationDiagnostic] = []
    for index, constraint in enumerate(contract.constraints.count):
        path = f"$.constraints.count[{index}].target"
        binding = binding_for_referent(contract, scene, path)
        if binding.grounded_ids:
            continue
        satisfied = (
            constraint.value == 0
            if constraint.comparison == "exactly"
            else constraint.value >= 0
            if constraint.comparison == "at_most"
            else constraint.value <= 0
        )
        if satisfied:
            continue
        soft = constraint.preference_source_text is not None
        diagnostics.append(
            _diagnostic(
                code="EMPTY_COUNT_UNAVOIDABLE_SOFT_VIOLATION"
                if soft
                else "EMPTY_COUNT_HARD_VIOLATION",
                checkpoint="scene_binding",
                category="GROUNDING",
                severity="WARNING" if soft else "BLOCK",
                node_or_path=path,
                assertion="empty COUNT is evaluated as zero",
                expected={
                    "comparison": constraint.comparison,
                    "value": constraint.value,
                },
                observed=0,
                detail={"candidate_set": []},
            )
        )
    return scene, diagnostics


def _scene_artifact_matches(
    working: Mapping[str, Any], scene: SceneGroundedContract
) -> bool:
    stored = working.get("scene_binding")
    return (
        isinstance(stored, Mapping)
        and stored.get("success") is True
        and _canonical_runtime_representation(stored.get("grounded_contract"))
        == _canonical_runtime_representation(scene.model_dump(mode="json"))
    )


def _store_scene_artifact(
    working: dict[str, Any],
    contract: SceneIndependentSemanticContract,
    label_grounding: Mapping[str, Any],
    scene: SceneGroundedContract,
) -> None:
    working["semantic_contract"] = contract.model_dump(mode="json")
    working["label_grounding"] = {
        key: copy.deepcopy(value)
        for key, value in label_grounding.items()
        if key not in {"stage1_gate_passed", "occurrence_count"}
    }
    working["scene_binding"] = {
        "executed": True,
        "success": True,
        "occurrences": [item.model_dump(mode="json") for item in scene.occurrences],
        "grounded_contract": scene.model_dump(mode="json"),
        "diagnostics": [],
    }


def _stale_lowering_parts(working: Mapping[str, Any], lowering: Any) -> list[str]:
    predicted = working.get("predicted")
    stored_lowering = working.get("lowering")
    checks = {
        "specification": (
            predicted.get("specification") if isinstance(predicted, Mapping) else None,
            lowering.specification,
        ),
        "grounded_constraints": (
            predicted.get("grounded_constraints")
            if isinstance(predicted, Mapping)
            else None,
            lowering.grounded_constraints,
        ),
        "dsl": (
            stored_lowering.get("dsl")
            if isinstance(stored_lowering, Mapping)
            else None,
            lowering.dsl,
        ),
    }
    return [
        name
        for name, (stored, current) in checks.items()
        if _canonical_runtime_representation(stored)
        != _canonical_runtime_representation(current)
    ]


def _canonical_runtime_representation(value: Any) -> Any:
    """Normalize only representationally empty optional collections for comparison."""
    if isinstance(value, Mapping):
        normalized = {}
        for key, child in value.items():
            child = _canonical_runtime_representation(child)
            if key in _EMPTY_OPTIONAL_COLLECTION_KEYS and child in (None, []):
                continue
            normalized[key] = child
        return normalized
    if isinstance(value, list):
        return [_canonical_runtime_representation(item) for item in value]
    return value


def _store_lowering_artifact(working: dict[str, Any], lowering: Any) -> None:
    working["lowering"] = {
        "executed": True,
        "dsl": lowering.dsl,
        "dsl_valid": True,
    }
    working["predicted"] = {
        "specification": lowering.specification,
        "grounded_constraints": lowering.grounded_constraints,
    }


def _lower_and_assert(
    contract: SceneIndependentSemanticContract,
    scene: SceneGroundedContract,
    robot_position: Mapping[str, Any],
    asg: ASG,
) -> tuple[Any | None, list[VerificationDiagnostic]]:
    try:
        lowering = lower_grounded_contract(contract, scene, robot_position, asg)
    except Exception as exc:  # noqa: BLE001 - checkpoint records backend failures
        carry_diagnostics = _carry_exception_diagnostics(contract, scene, exc)
        if carry_diagnostics:
            return None, carry_diagnostics
        return None, [
            _diagnostic(
                code="LOWERING_ASSERTION_FAILURE",
                checkpoint="route_lowering",
                category="LOWERING",
                severity="BLOCK",
                node_or_path="$.lowering",
                assertion="canonical commitments lower to a valid executable IR",
                expected="valid DSL",
                observed=str(exc),
                detail={"exception_type": type(exc).__name__},
            )
        ]
    carry_diagnostics = _carry_lowering_diagnostics(contract, scene, lowering)
    if carry_diagnostics:
        return lowering, carry_diagnostics
    return lowering, []


def _carry_exception_diagnostics(
    contract: SceneIndependentSemanticContract,
    scene: SceneGroundedContract,
    exc: Exception,
) -> list[VerificationDiagnostic]:
    message = str(exc)
    if message == "CARRY requires an explicit semantic endpoint":
        return [
            _diagnostic(
                code="CARRY_ENDPOINT_MISSING",
                checkpoint="route_lowering",
                category="COMMITMENT",
                severity="BLOCK",
                node_or_path=f"$.constraints.carry[{index}].scope.until",
                assertion="CARRY has an explicit semantic endpoint",
                expected="event boundary",
                observed=None,
                affected_commitments=[f"$.constraints.carry[{index}]"],
            )
            for index, item in enumerate(contract.constraints.carry)
            if item.scope is None or item.scope.until is None
        ]

    classifications = {
        "CARRY endpoint must resolve to a hard route event": (
            "CARRY_INVALID_ENDPOINT",
            "CARRY endpoint resolves to a hard route event",
        ),
        "CARRY endpoint cannot be navigation START": (
            "CARRY_INVALID_ENDPOINT",
            "CARRY endpoint is not navigation START",
        ),
        "CARRY acquisition boundary does not match a hard VIA for its payload": (
            "CARRY_ACQUISITION_BOUNDARY_MISMATCH",
            "CARRY acquisition boundary names a hard VIA with the same payload",
        ),
        "CARRY acquisition and endpoint must be distinct events": (
            "CARRY_ACQUISITION_EQUALS_ENDPOINT",
            "CARRY acquisition and endpoint are distinct events",
        ),
    }
    classified = classifications.get(message)
    if classified is not None:
        code, assertion = classified
        return [
            _diagnostic(
                code=code,
                checkpoint="route_lowering",
                category="COMMITMENT",
                severity="BLOCK",
                node_or_path="$.constraints.carry",
                assertion=assertion,
                expected="consistent CARRY interval semantics",
                observed=message,
                affected_commitments=[
                    f"$.constraints.carry[{index}]"
                    for index in range(len(contract.constraints.carry))
                ],
            )
        ]

    if message == "semantic width source must resolve to one object":
        invalid = []
        for index in range(len(contract.constraints.carry)):
            path = f"$.constraints.carry[{index}].payload"
            binding = binding_for_referent(contract, scene, path)
            if binding.asg_kind != "object" or len(binding.grounded_ids) != 1:
                invalid.append((index, binding.model_dump(mode="json")))
        return [
            _diagnostic(
                code="CARRY_PAYLOAD_INVALID",
                checkpoint="route_lowering",
                category="COMMITMENT",
                severity="BLOCK",
                node_or_path=f"$.constraints.carry[{index}].payload",
                assertion="CARRY payload resolves to one object",
                expected="one grounded object",
                observed=binding,
                affected_commitments=[f"$.constraints.carry[{index}]"],
            )
            for index, binding in invalid
        ]
    return []


def _carry_lowering_diagnostics(
    contract: SceneIndependentSemanticContract,
    scene: SceneGroundedContract,
    lowering: Any,
) -> list[VerificationDiagnostic]:
    issues: list[dict[str, Any]] = []
    elaborations = lowering.carry_elaborations
    if len(elaborations) != len(contract.constraints.carry):
        issues.append({
            "path": "$.constraints.carry",
            "expected": len(contract.constraints.carry),
            "observed": len(elaborations),
            "reason": "each CARRY requires one executable elaboration",
        })
    derived_names = [
        item.acquisition_name for item in elaborations if item.derived_acquisition
    ]
    if len(derived_names) != len(set(derived_names)):
        issues.append({
            "path": "$.lowering.carry_elaborations",
            "expected": "unique derived acquisition milestones",
            "observed": derived_names,
            "reason": "duplicate derived acquisition VIA",
        })
    for item in elaborations:
        path = f"$.constraints.carry[{item.carry_index}]"
        binding = binding_for_referent(contract, scene, item.acquisition_path)
        via_matches = [
            via
            for via in lowering.specification["via"]
            if via.get("name") == item.acquisition_name
        ]
        widths = [
            width
            for width in lowering.specification["width"]
            if width.get("derived_from") == "carry"
            and width.get("carry_index") == item.carry_index
        ]
        if binding.asg_kind != "object" or len(binding.grounded_ids) != 1:
            issues.append({
                "path": f"{path}.payload",
                "expected": "one grounded object",
                "observed": binding.model_dump(mode="json"),
                "reason": "CARRY payload is not a supported width source",
            })
        if len(via_matches) != 1:
            issues.append({
                "path": f"{path}.payload",
                "expected": item.acquisition_name,
                "observed": [via.get("name") for via in via_matches],
                "reason": "acquisition VIA is missing or duplicated",
            })
        expected_width = {
            "kind": "from_milestone",
            "milestone": item.acquisition_name,
            "after": item.acquisition_name,
            "until": item.until,
        }
        if len(widths) != 1 or any(
            widths[0].get(key) != value for key, value in expected_width.items()
        ):
            issues.append({
                "path": f"{path}.scope",
                "expected": expected_width,
                "observed": widths,
                "reason": "CARRY width interval does not match acquisition and endpoint",
            })
        expected_line = (
            f"ROBOT WIDTH FROM MILESTONE {item.acquisition_name} "
            f"AFTER {item.acquisition_name} UNTIL {item.until}"
        )
        if expected_line not in lowering.dsl:
            issues.append({
                "path": "$.lowering.dsl",
                "expected": expected_line,
                "observed": lowering.dsl,
                "reason": "stale width-from-object CARRY lowering",
            })
    return [
        _diagnostic(
            code="CARRY_ELABORATION_MISMATCH",
            checkpoint="route_lowering",
            category="LOWERING",
            severity="BLOCK",
            node_or_path=issue["path"],
            assertion="CARRY lowers to one compatible acquisition VIA and scoped width",
            expected=issue["expected"],
            observed=issue["observed"],
            detail={"reason": issue["reason"]},
        )
        for issue in issues
    ]


def _execute_and_assert(
    working: dict[str, Any],
    contract: SceneIndependentSemanticContract,
    label_grounding: Mapping[str, Any],
    scene: SceneGroundedContract,
    lowering: Any,
    robot_position: Mapping[str, Any],
    asg: ASG,
    *,
    recompute: bool,
    execution_backend: Callable | None = None,
) -> list[VerificationDiagnostic]:
    del contract
    if recompute:
        try:
            started = time.perf_counter()
            if execution_backend is None:
                route = solve_dsl_route(lowering.dsl, asg, profile=ExecutionProfile.FULL_HARD_SOFT)
            else:
                route = execution_backend(lowering.dsl, asg, profile=ExecutionProfile.FULL_HARD_SOFT,
                                          grounded_contract=lowering.grounded_constraints,
                                          robot_position=dict(robot_position))
            solver_seconds = time.perf_counter() - started
        except Exception as exc:  # noqa: BLE001 - classified below by known cause
            working["route"] = {}
            working["execution"] = {
                "executed": True,
                "solver_success": False,
                "profile": ExecutionProfile.FULL_HARD_SOFT.value,
                "error": f"{type(exc).__name__}: {exc}",
            }
            return [_solver_exception_diagnostic(exc)]
        working["semantic_contract"] = scene.semantic_contract
        working["label_grounding"] = {
            key: copy.deepcopy(value)
            for key, value in label_grounding.items()
            if key not in {"stage1_gate_passed", "occurrence_count"}
        }
        working["scene_binding"] = {
            "executed": True,
            "success": True,
            "occurrences": [item.model_dump(mode="json") for item in scene.occurrences],
            "grounded_contract": scene.model_dump(mode="json"),
            "diagnostics": [],
        }
        working["lowering"] = {"executed": True, "dsl": lowering.dsl, "dsl_valid": True}
        working["predicted"] = {
            "specification": lowering.specification,
            "grounded_constraints": lowering.grounded_constraints,
        }
        working["route"] = route
        found = route.get("status", "SAT") == "SAT" and bool(route.get("waypoints"))
        working["execution"] = {
            "executed": True,
            "solver_success": found,
            "profile": ExecutionProfile.FULL_HARD_SOFT.value,
        }
        working["stage_verification"] = {
            "json_valid": True,
            "schema_valid": True,
            "dsl_valid": True,
            "solver_success": found,
        }
        working["failure"] = (
            None
            if found
            else {
                "stage": "route",
                "reason": "NO_ROUTE",
                "detail": "solver returned no waypoints",
            }
        )
        working.setdefault("timing", {})["solver_seconds"] = solver_seconds
    route = working.get("route") if isinstance(working.get("route"), Mapping) else {}
    if not route.get("waypoints"):
        return [
            _diagnostic(
                code="SOLVER_UNKNOWN" if route.get("status") == "UNKNOWN" else "HARD_SAT_NOT_ESTABLISHED",
                checkpoint="route_verification",
                category="SOLVER",
                severity="BLOCK",
                node_or_path="$.route",
                assertion="a concrete route witnesses satisfiability of the fixed hard theory",
                expected="non-empty route",
                observed="no SAT witness",
                detail={"not_interpreted_as_proved_unsat": True},
            )
        ]
    fixed = lowering.grounded_constraints
    replay, errors = replayable_route(dict(route), fixed, asg, dict(robot_position))
    report = evaluate_grounded_route(asg, fixed, replay, dict(robot_position))
    hard_violations = [
        item
        for item in report.get("constraints", [])
        if not item.get("satisfied")
        and item.get("component")
        not in {"prefer", "prefer_order", "preference_optimality"}
    ]
    if errors or hard_violations:
        return [
            _diagnostic(
                code="BACKEND_ASSERTION_FAILURE",
                checkpoint="route_verification",
                category="EXECUTION",
                severity="BLOCK",
                node_or_path="$.route",
                assertion="returned route satisfies the same fixed grounded hard Contract",
                expected="all hard constraints satisfied",
                observed={"route_errors": errors, "hard_violations": hard_violations},
                detail={"fixed_grounded_specification": fixed},
            )
        ]
    return []


def _solver_exception_diagnostic(exc: Exception) -> VerificationDiagnostic:
    if isinstance(exc, _DETERMINISTIC_CONSISTENCY_ERRORS):
        code = "DETERMINISTIC_CONSISTENCY_FAILURE"
        category = "COMMITMENT"
        assertion = "the fixed hard specification is deterministically consistent"
        evidence = {
            "exception_type": type(exc).__name__,
            "classification": "HARD_SPEC_INCONSISTENT",
            "not_solver_unknown": True,
        }
    elif isinstance(exc, (SolveCancelledError, TimeoutError)):
        code = "SOLVER_UNKNOWN"
        category = "SOLVER"
        assertion = "hard satisfiability is established by a completed search"
        evidence = {
            "exception_type": type(exc).__name__,
            "not_interpreted_as_unsat": True,
        }
    else:
        code = "BACKEND_ERROR"
        category = "SOLVER"
        assertion = "the execution backend completes without an unexpected exception"
        evidence = {
            "exception_type": type(exc).__name__,
            "unexpected_runtime_exception": True,
        }
    return _diagnostic(
        code=code,
        checkpoint="route_verification",
        category=category,
        severity="BLOCK",
        node_or_path="$.execution",
        assertion=assertion,
        expected="SAT route",
        observed=str(exc),
        detail=evidence,
    )


def _executable_label_grounding(raw: Any, count: int) -> dict[str, Any]:
    label_grounding = copy.deepcopy(dict(raw)) if isinstance(raw, Mapping) else {}
    label_grounding["stage1_gate_passed"] = True
    label_grounding["occurrence_count"] = count
    return label_grounding


def _synthesized_label_grounding(
    collection: CommitmentCollection,
    determination: Determination,
) -> dict[str, Any]:
    bindings = dict(
        zip(
            determination.value("grounding.fully_determined_positions", ()),
            determination.value("grounding.fully_determined", ()),
        )
    )
    structural = dict(
        zip(
            determination.value("grounding.structurally_determined_positions", ()),
            determination.value("grounding.structurally_determined", ()),
        )
    )
    return {
        "stage1_gate_passed": True,
        "occurrence_count": len(collection.commitments),
        "all_grounding_items": [
            {
                "head_text": item.head_text,
                "source_text": item.source_text,
                "asg_kind": structural[position].asg_kind
                if position in structural
                else None,
            }
            for position, item in enumerate(collection.commitments)
        ],
        "final_bindings": [
            bindings[position].model_dump(mode="json")
            for position in range(len(collection.commitments))
        ],
        "validation": {
            "executed": True,
            "parse_success": True,
            "success": True,
            "diagnostics": [],
        },
        "llm_call_accounting": {
            "logical_calls": 0,
            "usage": {
                "llm_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "llm_latency_seconds": 0.0,
            },
            "attempt_count": 0,
            "attempts": [],
            "error": None,
            "semantic_retry_count": 0,
        },
    }


def _request_item(commitment: Any) -> Any:
    from reconav.method.grounding.contracts import GroundingItem

    return GroundingItem(
        head_text=commitment.head_text,
        source_text=commitment.source_text,
        asg_kind=None,
    )


def _with_runtime_facts(
    determination: Determination,
    *,
    contract: SceneIndependentSemanticContract,
    index: ContractIndex,
    scene: SceneGroundedContract | None,
    lowering: Any | None,
    working: Mapping[str, Any],
) -> Determination:
    facts: list[DeterminedFact] = []
    if scene is not None:
        occurrences = {item.path: item for item in scene.occurrences}
        facts.append(
            DeterminedFact(
                key="scene.candidate_sets",
                subject="$.scene_binding",
                value={
                    item.path: item.asg_binding.grounded_ids
                    for item in scene.occurrences
                },
                origin="PROPAGATION",
                rule="ASG_SCENE_BINDING",
            )
        )
        for item in scene.occurrences:
            facts.append(
                DeterminedFact(
                    key="grounding.referent",
                    subject=item.path,
                    value={
                        "head_text": item.head_text,
                        "source_text": item.source_text,
                        "committed_label": item.asg_binding.asg_label,
                        "resolved_asg_kind": item.asg_binding.asg_kind,
                        "scene_candidate_ids": item.asg_binding.grounded_ids,
                        "candidate_count": len(item.asg_binding.grounded_ids),
                    },
                    origin="PROPAGATION",
                    rule="ASG_SCENE_BINDING",
                )
            )
        for node in index.nodes:
            if node.kind == "EVENT":
                target = occurrences.get(f"{node.path}.target")
                facts.append(
                    DeterminedFact(
                        key="event.local",
                        subject=node.path,
                        value={
                            "handle": node.handle,
                            "event_type": node.role,
                            "target_referent_path": f"{node.path}.target",
                            "resolved_target": (
                                target.asg_binding.model_dump(mode="json")
                                if target is not None
                                else None
                            ),
                            "referenced_by": [
                                edge.from_handle
                                for edge in index.edges
                                if edge.kind == "EVENT_REFERENCE"
                                and edge.to_handle == node.handle
                            ],
                        },
                        origin="PROPAGATION",
                        rule="CONTRACT_EVENT_BINDING",
                    )
                )
    if lowering is not None:
        facts.append(
            DeterminedFact(
                key="lowering.product",
                subject="$.lowering",
                value={
                    "dsl": lowering.dsl,
                    "specification": lowering.specification,
                    "grounded_constraints": lowering.grounded_constraints,
                },
                origin="PROPAGATION",
                rule="CONTRACT_LOWERING",
            )
        )
        for node in index.nodes:
            if node.kind != "CONSTRAINT":
                continue
            canonical = _value_at_path(contract.model_dump(mode="json"), node.path)
            operands = {
                item.path: item.asg_binding.model_dump(mode="json")
                for item in scene.occurrences
                if item.path.startswith(node.path + ".")
            } if scene is not None else {}
            event_references = [
                {
                    "path": edge.path,
                    "target_source_text": edge.target_source_text,
                    "resolution": edge.resolution,
                    "resolved_event_handle": edge.to_handle,
                }
                for edge in index.edges
                if edge.kind == "EVENT_REFERENCE" and edge.from_handle == node.handle
            ]
            value = {
                "handle": node.handle,
                "constraint_type": node.role,
                "canonical": canonical,
                "grounded_operands": operands,
                "event_references": event_references,
            }
            carry_match = re.fullmatch(r"\$\.constraints\.carry\[(\d+)]", node.path)
            if carry_match:
                carry_index = int(carry_match.group(1))
                elaboration = next(
                    (
                        item
                        for item in lowering.carry_elaborations
                        if item.carry_index == carry_index
                    ),
                    None,
                )
                value["carry_elaboration"] = (
                    {
                        "acquisition_name": elaboration.acquisition_name,
                        "acquisition_path": elaboration.acquisition_path,
                        "until": elaboration.until,
                        "derived_acquisition": elaboration.derived_acquisition,
                    }
                    if elaboration is not None
                    else None
                )
                value["derived_operational_widths"] = [
                    item
                    for item in lowering.grounded_constraints.get("width", [])
                    if item.get("source") == "carry"
                    or item.get("carry_index") == carry_index
                ]
            facts.append(
                DeterminedFact(
                    key="constraint.lowering",
                    subject=node.path,
                    value=value,
                    origin="PROPAGATION",
                    rule="CONSTRAINT_LOCAL_LOWERING",
                )
            )
    execution = working.get("execution")
    route = working.get("route")
    if lowering is None:
        solver_status = "NOT_RUN"
    elif isinstance(execution, Mapping) and (
        execution.get("error") or execution.get("solver_status") == "UNKNOWN"
    ):
        solver_status = "UNKNOWN"
    elif (
        isinstance(execution, Mapping)
        and execution.get("solver_success") is True
        and isinstance(route, Mapping)
        and route.get("waypoints")
    ):
        solver_status = "SAT_WITNESS_FOUND"
    else:
        solver_status = "NO_SAT_WITNESS"
    facts.append(
        DeterminedFact(
            key="solver.execution_status",
            subject="$.execution",
            value={
                "status": solver_status,
                "solver_success": (
                    execution.get("solver_success")
                    if isinstance(execution, Mapping)
                    else None
                ),
                "waypoint_count": (
                    len(route.get("waypoints", []))
                    if isinstance(route, Mapping)
                    else 0
                ),
                "error": execution.get("error") if isinstance(execution, Mapping) else None,
                "not_interpreted_as_proved_unsat": solver_status == "NO_SAT_WITNESS",
            },
            origin="SOLVER",
            rule="DSL_ROUTE_SEARCH",
        )
    )
    return determination.extended(facts=facts)


def _contract_index_facts(index: ContractIndex) -> list[DeterminedFact]:
    facts = [
        DeterminedFact(
            key="contract.node",
            subject=node.path,
            value=node.model_dump(mode="json"),
            origin="PROPAGATION",
            rule="CONTRACT_INDEX",
        )
        for node in index.nodes
    ]
    for edge in index.edges:
        facts.append(
            DeterminedFact(
                key="contract.reference",
                subject=edge.path,
                value=edge.model_dump(mode="json"),
                origin="PROPAGATION",
                rule="CONTRACT_INDEX_RESOLUTION",
            )
        )
        if edge.kind != "EVENT_REFERENCE" or not edge.target_source_offsets:
            continue
        overlapping = [
            node.handle
            for node in index.nodes
            if any(
                left < target_right and target_left < right
                for left, right in node.source_offsets
                for target_left, target_right in edge.target_source_offsets
            )
        ]
        facts.append(
            DeterminedFact(
                key="contract.source_overlap",
                subject=edge.path,
                value={
                    "target_source_text": edge.target_source_text,
                    "overlapping_handles": overlapping,
                },
                origin="PROPAGATION",
                rule="LEXICAL_SOURCE_OCCURRENCE_OVERLAP",
            )
        )
    return facts


def _value_at_path(value: Any, path: str) -> Any:
    current = value
    for field, index in re.findall(r"(?:^\$\.|\.)([^.\[]+)(?:\[(\d+)])?", path):
        if not isinstance(current, Mapping) or field not in current:
            return None
        current = current[field]
        if index:
            if not isinstance(current, list) or int(index) >= len(current):
                return None
            current = current[int(index)]
    return copy.deepcopy(current)


def _diagnostic(
    *,
    code: str,
    checkpoint: Literal["contract_validation", "label_grounding", "scene_binding", "route_lowering", "route_verification"],
    category: Literal[
        "ARTIFACT", "COMMITMENT", "GROUNDING", "LOWERING", "SOLVER", "EXECUTION"
    ],
    severity: Literal["WARNING", "BLOCK"],
    node_or_path: str,
    assertion: str,
    expected: Any = None,
    observed: Any = None,
    supporting_facts: list[str] | None = None,
    detail: dict[str, Any] | None = None,
    affected_commitments: list[str] | None = None,
    repairable: bool = False,
) -> VerificationDiagnostic:
    needs_backend_fix = severity == "BLOCK" and code in {
        "LOWERING_ASSERTION_FAILURE",
        "CARRY_ELABORATION_MISMATCH",
        "SOLVER_UNKNOWN",
        "BACKEND_ERROR",
        "BACKEND_ASSERTION_FAILURE",
        "UPSTREAM_INFRASTRUCTURE_FAILURE",
    }
    return VerificationDiagnostic(
        code=code,
        checkpoint=checkpoint,
        category=category,
        severity=severity,
        node_or_path=node_or_path,
        assertion=assertion,
        expected=expected,
        observed=observed,
        supporting_facts=supporting_facts or [],
        detail=detail or {},
        affected_commitments=affected_commitments or [],
        repairable=repairable,
        pre_status=severity,
        post_status=None,
        needs_backend_fix=needs_backend_fix,
    )


def _finish(
    *,
    working: dict[str, Any],
    index: ContractIndex,
    determination: Determination,
    before: list[VerificationDiagnostic],
    after: list[VerificationDiagnostic],
    iterations: int,
    backend_assertion_pass: bool | None = None,
) -> VerificationOutcome:
    before_status = status_for(before)
    after_status = status_for(after)
    before = [item.model_copy(update={"post_status": after_status}) for item in before]
    after = [
        item.model_copy(
            update={
                "pre_status": before_status,
                "post_status": after_status,
            }
        )
        for item in after
    ]
    invalidated = sorted(
        {value for action in determination.actions for value in action.invalidates}
    )
    if after_status == "BLOCK":
        invalidated = sorted(
            {*invalidated, *_invalidate_blocked_descendants(working, after)}
        )
    label_grounding_reexecuted = any(
        "ASG_INSTANCE_BINDING" in action.invalidates
        for action in determination.actions
    )
    return VerificationOutcome(
        contract_index=index,
        determination=determination,
        diagnostics_before_repair=before,
        diagnostics_after_repair=after,
        initial_status=before_status,
        final_status=after_status,
        verification_accepted=after_status != "BLOCK",
        backend_assertion_pass=backend_assertion_pass,
        needs_backend_fix=any(item.needs_backend_fix for item in after),
        repair_iterations=iterations,
        verification_llm_calls=0,
        additional_forward_llm_calls=0,
        label_grounding_reexecution_count=int(label_grounding_reexecuted),
        new_label_grounding_residual_llm_calls=0,
        invalidated_descendants=invalidated,
        prediction_artifact=working,
    )


def _blocked_outcome(
    working: dict[str, Any], diagnostics: list[VerificationDiagnostic]
) -> VerificationOutcome:
    final = [item.model_copy(update={"post_status": "BLOCK"}) for item in diagnostics]
    invalidated = _invalidate_blocked_descendants(working, final)
    return VerificationOutcome(
        contract_index=None,
        determination=Determination(),
        diagnostics_before_repair=final,
        diagnostics_after_repair=final,
        initial_status="BLOCK",
        final_status="BLOCK",
        verification_accepted=False,
        backend_assertion_pass=None,
        needs_backend_fix=any(item.needs_backend_fix for item in final),
        repair_iterations=0,
        verification_llm_calls=0,
        additional_forward_llm_calls=0,
        label_grounding_reexecution_count=0,
        new_label_grounding_residual_llm_calls=0,
        invalidated_descendants=invalidated,
        prediction_artifact=working,
    )


def _invalidate_blocked_descendants(
    working: dict[str, Any], diagnostics: list[VerificationDiagnostic]
) -> list[str]:
    order = {"contract_validation": 0, "label_grounding": 1, "scene_binding": 2, "route_lowering": 3, "route_verification": 4}
    checkpoints = [
        item.checkpoint for item in diagnostics if item.severity == "BLOCK"
    ]
    earliest = min(checkpoints, key=order.__getitem__)
    invalidated: list[str] = []
    if order[earliest] <= order["scene_binding"]:
        working["scene_binding"] = {
            "executed": False,
            "success": False,
            "occurrences": [],
            "diagnostics": [],
        }
        invalidated.extend(["ASG_INSTANCE_BINDING", "GROUNDED_CONTRACT"])
    if order[earliest] <= order["route_lowering"]:
        working["lowering"] = {"executed": False, "dsl_valid": False}
        working["predicted"] = {}
        invalidated.append("DSL_LOWERING")
    working["execution"] = {"executed": False, "solver_success": False}
    working["route"] = {}
    working["stage_verification"] = {
        "json_valid": earliest != "contract_validation",
        "schema_valid": earliest != "contract_validation",
        "dsl_valid": order[earliest] > order["route_lowering"],
        "solver_success": False,
    }
    working["failure"] = {
        "stage": "verification",
        "reason": "BLOCK",
        "detail": [
            item.code for item in diagnostics if item.severity == "BLOCK"
        ],
    }
    timing = working.get("timing")
    if isinstance(timing, dict):
        timing.pop("solver_seconds", None)
    invalidated.extend(
        ["SOLVER_RESULT", "ROUTE", "WAYPOINTS", "EVALUATOR_PREDICTION"]
    )
    return sorted(set(invalidated))


def _backend_assertion_pass(
    diagnostics: list[VerificationDiagnostic], working: Mapping[str, Any]
) -> bool | None:
    if any(item.code == "BACKEND_ASSERTION_FAILURE" for item in diagnostics):
        return False
    route = working.get("route")
    if (
        isinstance(route, Mapping)
        and route.get("waypoints")
        and not any(item.severity == "BLOCK" for item in diagnostics)
    ):
        return True
    return None


def _deduplicate(items: list[VerificationDiagnostic]) -> list[VerificationDiagnostic]:
    seen: set[tuple[str, str]] = set()
    output: list[VerificationDiagnostic] = []
    for item in items:
        key = item.code, item.node_or_path
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output
