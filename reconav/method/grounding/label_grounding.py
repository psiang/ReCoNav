"""Occurrence-level semantic grounding from Stage-1 nominals to ASG labels."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from reconav.method.determination import Determination, DeterminedFact
from reconav.method.grounding.contracts import (
    ASGKind,
    ASGVocabulary,
    GroundingBinding,
    GroundingItem,
    GroundingRequest,
    GroundingResult,
)
from reconav.llm.shared import LLMClient
from reconav.method.semantic.contracts import SceneIndependentSemanticContract

PROMPT_ROOT = Path(__file__).resolve().parents[2] / "prompts" / "semantic_grounding"
COMMON_PREFIX_PATH = PROMPT_ROOT / "common_prefix.txt"
RECONAV_POLICY_PATH = PROMPT_ROOT / "reconav_policy.txt"
COMMON_SUFFIX_PATH = PROMPT_ROOT / "common_suffix.txt"
ALL_KINDS: frozenset[ASGKind] = frozenset({"object", "region", "portal", "level"})
_DETERMINERS = frozenset({"a", "an", "the", "any", "every", "each", "all"})
_SPACE_RE = re.compile(r"\s+")
_LEVEL_WORDS = frozenset({"floor", "level", "story", "storey", "upstairs", "downstairs"})
# Public ontology heads denote the entire kind, independently of instance labels.
# A head may ALSO match a category of another kind (floor surfaces vs storeys).
# Keep both readings; lexical equality must not silently resolve that ambiguity.
GENERIC_KIND_HEADS = {
    "object": frozenset({"object"}),
    "region": frozenset({"region", "area", "space"}),
    "portal": frozenset(),
    "level": frozenset({"level", "floor", "storey", "story"}),
}
_QUALIFIER_SIGNATURES: dict[str, tuple[tuple[ASGKind, ASGKind], ...]] = {
    "in": (
        ("object", "region"),
        ("object", "level"),
        ("region", "level"),
        ("portal", "region"),
    ),
    "near": (("object", "object"),),
    "nearest_to": (("object", "object"), ("object", "start")),
    "farthest_from": (("object", "object"), ("object", "start")),
    "between": (("portal", "region"),),
}


def _prompt_component(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"empty grounding prompt component: {path}")
    return value


COMMON_PREFIX = _prompt_component(COMMON_PREFIX_PATH)
RECONAV_POLICY = _prompt_component(RECONAV_POLICY_PATH)
COMMON_SUFFIX = _prompt_component(COMMON_SUFFIX_PATH)
RECONAV_PROMPT = "\n\n".join((COMMON_PREFIX, RECONAV_POLICY, COMMON_SUFFIX))


@dataclass(frozen=True)
class NominalCommitment:
    head_text: str
    source_text: str
    path: str
    owner_path: str
    possible_kinds: frozenset[ASGKind]
    structural_reasons: tuple[str, ...]
    nominal_source_text: str | None = None


@dataclass(frozen=True)
class GroundingDiagnostic:
    code: str
    head_text: str
    path: str
    detail: str


@dataclass(frozen=True)
class KindRelation:
    owner_position: int
    reference_position: int
    relation: str


@dataclass(frozen=True)
class CommitmentCollection:
    commitments: tuple[NominalCommitment, ...]
    diagnostics: tuple[GroundingDiagnostic, ...]
    relations: tuple[KindRelation, ...] = ()


@dataclass(frozen=True)
class SemanticGroundingRun:
    request_items: tuple[GroundingItem, ...]
    learned_bindings: tuple[GroundingBinding, ...]
    final_bindings: tuple[GroundingBinding, ...]
    raw_response: str | None
    parse_success: bool
    validation_success: bool
    validation_diagnostics: tuple[str, ...]
    llm_calls: int
    expected_binding_count: int
    response_schema_sha256: str | None
    determination: Determination | None = None


@dataclass(frozen=True)
class GroundingConfig:
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: int = 0
    enable_thinking: bool = False


def vocabulary_from_entity_payload(payload: dict[str, Any]) -> ASGVocabulary:
    """Build the kind-separated semantic vocabulary from ASG entity JSON."""

    def labels(key: str) -> list[str]:
        entries = payload.get(key)
        if not isinstance(entries, list):
            raise ValueError(f"ASG entity payload field {key!r} must be a list")
        values = {
            str(item["label"])
            for item in entries
            if isinstance(item, dict) and item.get("label")
        }
        return sorted(values)

    return ASGVocabulary(
        **{kind: sorted(set(labels(key)) | {"any"} | ({"room"} if kind == "region" else
                      {"portal"} if kind == "portal" else set())) for kind, key in
           (("object", "objects"), ("region", "regions"),
            ("portal", "portals"), ("level", "levels"))}
    )


def collect_nominal_commitments(
    contract: SceneIndependentSemanticContract,
) -> CommitmentCollection:
    """Collect each nominal AST occurrence independently in source order."""

    commitments: list[NominalCommitment] = []
    diagnostics: list[GroundingDiagnostic] = []
    relations: list[KindRelation] = []

    def append_occurrence(
        *,
        head_text: str,
        source_text: str,
        path: str,
        owner_path: str,
        requirements: list[tuple[frozenset[ASGKind], str]],
        nominal_source_text: str,
    ) -> int:
        possible = set(ALL_KINDS)
        for kinds, _reason in requirements:
            possible.intersection_update(kinds)
        reasons = tuple(dict.fromkeys(reason for _kinds, reason in requirements))
        commitment = NominalCommitment(
            head_text=head_text,
            source_text=source_text,
            path=path,
            owner_path=owner_path,
            possible_kinds=frozenset(possible),
            structural_reasons=reasons,
            nominal_source_text=nominal_source_text,
        )
        commitments.append(commitment)
        if not possible:
            diagnostics.append(
                GroundingDiagnostic(
                    code="INCONSISTENT_KIND_REQUIREMENT",
                    head_text=head_text,
                    path=path,
                    detail="; ".join(reasons),
                )
            )
        return len(commitments) - 1

    def visit_reference(
        value: Any,
        path: str,
        owner_path: str,
        owner_source_text: str,
        required: frozenset[ASGKind],
        reason: str,
    ) -> int | None:
        if value.kind == "start":
            return None
        if value.kind == "coreference":
            return visit_reference(
                value.antecedent,
                f"{path}.antecedent",
                owner_path,
                owner_source_text,
                required,
                reason,
            )
        return visit_referent(
            value, path,
            required_kinds=required,
            reason=reason,
            owner_path=owner_path,
            owner_source_text=owner_source_text,
        )

    def visit_referent(
        value: Any,
        path: str,
        required: ASGKind | None = None,
        reason: str | None = None,
        *,
        required_kinds: frozenset[ASGKind] | None = None,
        owner_path: str | None = None,
        owner_source_text: str | None = None,
    ) -> int | None:
        if value.kind == "start":
            return
        if value.kind == "coreference":
            return visit_referent(value.antecedent, f"{path}.antecedent", required, reason,
                                  required_kinds=required_kinds, owner_path=owner_path,
                                  owner_source_text=owner_source_text)

        owner_requirements: list[tuple[frozenset[ASGKind], str]] = []
        if required is not None and reason is not None:
            owner_requirements.append((frozenset({required}), reason))
        if required_kinds is not None and reason is not None:
            owner_requirements.append((required_kinds, reason))
        qualifiers = value.qualifiers or []
        for index, qualifier in enumerate(qualifiers):
            signatures = _QUALIFIER_SIGNATURES[qualifier.relation]
            owner_kinds = frozenset(owner_kind for owner_kind, _reference_kind in signatures)
            owner_requirements.append(
                (
                    owner_kinds,
                    f"{path}.qualifiers[{index}]: {qualifier.relation.upper()} owner is one of "
                    + ", ".join(sorted(owner_kinds)),
                )
            )
        owner_position = append_occurrence(
            head_text=value.head_text,
            nominal_source_text=value.source_text,
            source_text=owner_source_text or value.source_text,
            path=path,
            owner_path=owner_path or path,
            requirements=owner_requirements,
        )
        for q_index, qualifier in enumerate(qualifiers):
            q_path = f"{path}.qualifiers[{q_index}]"
            signatures = _QUALIFIER_SIGNATURES[qualifier.relation]
            reference_kinds = frozenset(reference_kind for _owner_kind, reference_kind in signatures)
            reference_reason = (
                f"{q_path}: {qualifier.relation.upper()} reference is one of "
                + ", ".join(sorted(reference_kinds))
            )
            for r_index, reference in enumerate(qualifier.references):
                reference_position = visit_reference(
                    reference,
                    f"{q_path}.references[{r_index}]",
                    path,
                    value.source_text,
                    reference_kinds,
                    reference_reason,
                )
                if reference_position is not None:
                    relations.append(
                        KindRelation(
                            owner_position=owner_position,
                            reference_position=reference_position,
                            relation=qualifier.relation,
                        )
                    )
        return owner_position

    for index, event in enumerate(contract.route.via_events):
        visit_referent(event.target, f"$.route.via_events[{index}].target",
                       owner_source_text=event.source_text)
    visit_referent(contract.route.goal.target, "$.route.goal.target",
                   owner_source_text=contract.route.goal.source_text)
    for index, constraint in enumerate(contract.constraints.avoid):
        visit_referent(constraint.target, f"$.constraints.avoid[{index}].target",
                       owner_source_text=constraint.source_text)
    for index, constraint in enumerate(contract.constraints.count):
        visit_referent(constraint.target, f"$.constraints.count[{index}].target",
                       owner_source_text=constraint.source_text)
    for index, constraint in enumerate(contract.constraints.carry):
        visit_referent(
            constraint.payload,
            f"$.constraints.carry[{index}].payload",
            "object",
            "CARRY payload is object",
            owner_source_text=constraint.source_text,
        )
    for index, constraint in enumerate(contract.constraints.robot_width):
        if hasattr(constraint, "reference"):
            visit_referent(
                constraint.reference,
                f"$.constraints.robot_width[{index}].reference",
                "object",
                "reference-derived width source is object",
                owner_source_text=constraint.source_text,
            )
    return CommitmentCollection(tuple(commitments), tuple(diagnostics), tuple(relations))


def determine_groundings(
    collection: CommitmentCollection,
    vocabulary: ASGVocabulary,
) -> Determination:
    """Discharge only structural and exact conservative lexical facts."""

    possible_kinds = _propagated_kind_domains(collection, vocabulary)
    structural_positions: list[int] = []
    structural_items: list[GroundingItem] = []
    pre_llm_kind_positions: list[int] = []
    full_positions: list[int] = []
    full: list[GroundingBinding] = []
    partial: list[GroundingItem] = []
    residual_positions: list[int] = []
    residual: list[GroundingItem] = []
    inconsistent_paths = {item.path for item in collection.diagnostics}
    for position, commitment in enumerate(collection.commitments):
        possible = possible_kinds[position]
        if commitment.path in inconsistent_paths or not possible:
            continue
        structural_kind = (
            next(iter(possible))
            if len(possible) == 1
            else None
        )
        if structural_kind is not None:
            structural_positions.append(position)
            structural_items.append(
                GroundingItem(
                    head_text=commitment.head_text,
                    source_text=commitment.source_text,
                    asg_kind=structural_kind,
                )
            )
        binding = _canonical_binding(commitment, vocabulary, possible)
        if binding is not None:
            full_positions.append(position)
            full.append(binding)
            pre_llm_kind_positions.append(position)
            continue
        item = GroundingItem(
            head_text=commitment.head_text,
            source_text=commitment.source_text,
            asg_kind=structural_kind,
        )
        residual_positions.append(position)
        residual.append(item)
        if structural_kind is not None:
            partial.append(item)
            pre_llm_kind_positions.append(position)
    facts: list[DeterminedFact] = []
    for position, commitment in enumerate(collection.commitments):
        subject = commitment.path
        commit_key = f"grounding.{position}.commitment"
        kinds_key = f"grounding.{position}.possible_kinds"
        facts.extend(
            [
                DeterminedFact(
                    key=commit_key,
                    subject=subject,
                    value={
                        "head_text": commitment.head_text,
                        "source_text": commitment.source_text,
                        "owner_path": commitment.owner_path,
                    },
                    origin="COMMIT",
                    rule="ACCEPTED_NOMINAL_OCCURRENCE",
                ),
                DeterminedFact(
                    key=kinds_key,
                    subject=subject,
                    value=sorted(possible_kinds[position]),
                    origin="PROPAGATION",
                    rule="FIXPOINT_SIGNATURE_PROPAGATION",
                ),
            ]
        )
        if position in structural_positions:
            item = structural_items[structural_positions.index(position)]
            facts.append(
                DeterminedFact(
                    key=f"grounding.{position}.structural_binding",
                    subject=subject,
                    value=item,
                    origin="PROPAGATION",
                    rule="UNIQUE_STRUCTURAL_KIND",
                )
            )
        if position in full_positions:
            binding = full[full_positions.index(position)]
            facts.append(
                DeterminedFact(
                    key=f"grounding.{position}.canonical_binding",
                    subject=subject,
                    value=binding,
                    origin="PROPAGATION",
                    rule=("CANONICAL_KIND_HEAD" if binding.asg_label == "any"
                          else "CANONICAL_LEXICAL_EQUALITY"),
                )
            )
        facts.append(
            DeterminedFact(
                key=f"grounding.{position}.status",
                subject=subject,
                value="FULLY_DETERMINED"
                if position in full_positions
                else "RESIDUAL",
                origin="PROPAGATION",
                rule="GROUNDING_DETERMINATION_CLASSIFICATION",
            )
        )
    for relation in collection.relations:
        owner_kinds = possible_kinds[relation.owner_position]
        reference_kinds = possible_kinds[relation.reference_position]
        if len(owner_kinds) != 1 or len(reference_kinds) != 1:
            continue
        signature = (next(iter(owner_kinds)), next(iter(reference_kinds)))
        if signature not in _QUALIFIER_SIGNATURES[relation.relation]:
            continue
        facts.append(
            DeterminedFact(
                key=(
                    f"grounding.relation.{relation.owner_position}."
                    f"{relation.reference_position}.{relation.relation}"
                ),
                subject=collection.commitments[relation.owner_position].owner_path,
                value={
                    "owner_kind": signature[0],
                    "reference_kind": signature[1],
                },
                origin="PROPAGATION",
                rule=_relation_rule(relation.relation, signature),
            )
        )
    aggregate_values = {
        "grounding.structurally_determined_positions": tuple(structural_positions),
        "grounding.structurally_determined": tuple(structural_items),
        "grounding.pre_llm_kind_determined_positions": tuple(
            pre_llm_kind_positions
        ),
        "grounding.fully_determined_positions": tuple(full_positions),
        "grounding.fully_determined": tuple(full),
        "grounding.partial_kind_only": tuple(partial),
        "grounding.residual_positions": tuple(residual_positions),
        "grounding.residual_items": tuple(residual),
    }
    facts.extend(
        DeterminedFact(
            key=key,
            subject="$.label_grounding",
            value=value,
            origin="PROPAGATION",
            rule="GROUNDING_DETERMINATION_SUMMARY",
        )
        for key, value in aggregate_values.items()
    )
    return Determination(facts=facts)


class DetermineFirstSemanticGrounder:
    def __init__(self, client: LLMClient, config: GroundingConfig | None = None) -> None:
        self.client = client
        self.config = config or GroundingConfig()

    def ground(
        self,
        contract: SceneIndependentSemanticContract,
        vocabulary: ASGVocabulary,
        *,
        collection: CommitmentCollection | None = None,
    ) -> SemanticGroundingRun:
        collection = collection or collect_nominal_commitments(contract)
        determination = determine_groundings(collection, vocabulary)
        residual_items = tuple(
            determination.value("grounding.residual_items", ())
        )
        residual_positions = tuple(
            determination.value("grounding.residual_positions", ())
        )
        fully_determined_positions = tuple(
            determination.value("grounding.fully_determined_positions", ())
        )
        fully_determined = tuple(
            determination.value("grounding.fully_determined", ())
        )
        structural_positions = tuple(
            determination.value("grounding.structurally_determined_positions", ())
        )
        structural_items = tuple(
            determination.value("grounding.structurally_determined", ())
        )
        learned, raw, parse_success, diagnostics, calls, schema_hash = self._call(
            vocabulary,
            residual_items,
            RECONAV_PROMPT,
        )
        final = _positional_merge(
            len(collection.commitments),
            fully_determined_positions,
            fully_determined,
            residual_positions,
            learned,
        )
        structural_by_position = dict(
            zip(structural_positions, structural_items)
        )
        expected_items = tuple(
            GroundingItem(
                head_text=item.head_text,
                source_text=item.source_text,
                asg_kind=(
                    structural_by_position[position].asg_kind
                    if position in structural_by_position
                    else None
                ),
            )
            for position, item in enumerate(collection.commitments)
        )
        final_diagnostics = list(diagnostics)
        final_diagnostics.extend(
            f"{item.code}[path={item.path}]: {item.head_text}: {item.detail}"
            for item in collection.diagnostics
        )
        final_diagnostics.extend(validate_bindings(expected_items, final, vocabulary))
        final_diagnostics.extend(validate_relation_signatures(collection, final))
        return SemanticGroundingRun(
            request_items=residual_items,
            learned_bindings=learned,
            final_bindings=final,
            raw_response=raw,
            parse_success=parse_success,
            validation_success=not final_diagnostics,
            validation_diagnostics=tuple(dict.fromkeys(final_diagnostics)),
            llm_calls=calls,
            expected_binding_count=len(residual_items),
            response_schema_sha256=schema_hash,
            determination=determination,
        )

    def _call(
        self,
        vocabulary: ASGVocabulary,
        items: tuple[GroundingItem, ...],
        prompt: str,
    ) -> tuple[
        tuple[GroundingBinding, ...],
        str | None,
        bool,
        tuple[str, ...],
        int,
        str | None,
    ]:
        if not items:
            return (), None, True, (), 0, None
        visible_vocabulary = _visible_vocabulary(vocabulary, items)
        request = GroundingRequest(items=list(items), asg_vocabulary=visible_vocabulary)
        response_format = grounding_response_format(len(items))
        raw = self.client.generate(
            system_prompt=prompt,
            user_prompt=json.dumps(request.model_dump(mode="json"), ensure_ascii=False),
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            min_p=self.config.min_p,
            presence_penalty=self.config.presence_penalty,
            repetition_penalty=self.config.repetition_penalty,
            seed=self.config.seed,
            enable_thinking=self.config.enable_thinking,
            response_format=response_format,
        )
        parsed, parse_error = parse_grounding_result(raw)
        schema_hash = _json_sha256(response_format)
        if parsed is None:
            return (), raw, False, (f"SCHEMA_VALIDATION_FAILED: {parse_error}",), 1, schema_hash
        diagnostics = validate_bindings(items, parsed.bindings, visible_vocabulary)
        return tuple(parsed.bindings), raw, True, diagnostics, 1, schema_hash




def parse_grounding_result(raw: str) -> tuple[GroundingResult | None, str | None]:
    try:
        candidate = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    try:
        return GroundingResult.model_validate(candidate), None
    except ValidationError as exc:
        return None, str(exc)


def validate_bindings(
    requested: Iterable[GroundingItem],
    bindings: Iterable[GroundingBinding],
    vocabulary: ASGVocabulary,
) -> tuple[str, ...]:
    """Validate one response binding against each request position."""

    requested_list = list(requested)
    binding_list = list(bindings)
    diagnostics: list[str] = []
    if len(binding_list) != len(requested_list):
        diagnostics.append(
            f"CARDINALITY_MISMATCH: expected {len(requested_list)}, got {len(binding_list)}"
        )
    for index, (item, binding) in enumerate(zip(requested_list, binding_list)):
        if binding.head_text != item.head_text:
            diagnostics.append(
                f"POSITIONAL_HEAD_MISMATCH[index={index}]: "
                f"expected {item.head_text!r}, got {binding.head_text!r}"
            )
        if item.asg_kind is not None and binding.asg_kind != item.asg_kind:
            diagnostics.append(
                f"KIND_HINT_VIOLATION[index={index}]: "
                f"expected {item.asg_kind}, got {binding.asg_kind}"
            )
        if binding.asg_kind not in ALL_KINDS:
            diagnostics.append(
                f"INVALID_KIND[index={index}]: {binding.asg_kind}"
            )
            continue
        if binding.asg_label not in _labels_for_kind(vocabulary, binding.asg_kind):
            diagnostics.append(
                f"UNSUPPORTED_LABEL[index={index}]: "
                f"{binding.asg_kind}/{binding.asg_label}"
            )
    return tuple(diagnostics)


def validate_relation_signatures(
    collection: CommitmentCollection,
    bindings: Iterable[GroundingBinding],
) -> tuple[str, ...]:
    """Validate committed qualifier bindings against canonical relation signatures."""

    binding_list = list(bindings)
    diagnostics: list[str] = []
    for relation in collection.relations:
        if max(relation.owner_position, relation.reference_position) >= len(binding_list):
            continue
        owner_kind = binding_list[relation.owner_position].asg_kind
        reference_kind = binding_list[relation.reference_position].asg_kind
        signature = (owner_kind, reference_kind)
        signatures = _QUALIFIER_SIGNATURES[relation.relation]
        if signature in signatures:
            continue
        expected = ", ".join(f"{owner}/{reference}" for owner, reference in signatures)
        diagnostics.append(
            "RELATION_SIGNATURE_VIOLATION"
            f"[owner_index={relation.owner_position},"
            f"reference_index={relation.reference_position},"
            f"relation={relation.relation.upper()}]: "
            f"expected one of {expected}; got {owner_kind}/{reference_kind}"
        )
    return tuple(diagnostics)


def grounding_response_json_schema(expected_count: int) -> dict[str, Any]:
    if expected_count < 0:
        raise ValueError("expected_count must not be negative")
    schema = GroundingResult.model_json_schema()
    bindings = schema["properties"]["bindings"]
    bindings["minItems"] = expected_count
    bindings["maxItems"] = expected_count
    return schema


def grounding_response_format(expected_count: int) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "semantic_grounding_result",
            "strict": True,
            "schema": grounding_response_json_schema(expected_count),
        },
    }


def _propagated_kind_domains(
    collection: CommitmentCollection,
    vocabulary: ASGVocabulary,
) -> tuple[frozenset[ASGKind], ...]:
    domains = [set(item.possible_kinds) for item in collection.commitments]
    for position, commitment in enumerate(collection.commitments):
        matches = _canonical_candidates(commitment, vocabulary, domains[position])
        if matches:
            domains[position].intersection_update(kind for kind, _label in matches)
    changed = True
    while changed:
        changed = False
        for relation in collection.relations:
            owner = domains[relation.owner_position]
            reference = domains[relation.reference_position]
            owner_allowed = {
                owner_kind
                for owner_kind, reference_kind in _QUALIFIER_SIGNATURES[relation.relation]
                if owner_kind in owner and reference_kind in reference
            }
            reference_allowed = {
                reference_kind
                for owner_kind, reference_kind in _QUALIFIER_SIGNATURES[relation.relation]
                if owner_kind in owner and reference_kind in reference
            }
            if owner_allowed != owner:
                domains[relation.owner_position] = owner_allowed
                changed = True
            if reference_allowed != reference:
                domains[relation.reference_position] = reference_allowed
                changed = True
    return tuple(frozenset(domain) for domain in domains)


def _relation_rule(relation: str, signature: tuple[ASGKind, ASGKind]) -> str:
    if relation == "in":
        return {
            ("object", "region"): "OBJECT_IN_REGION",
            ("object", "level"): "OBJECT_IN_LEVEL",
            ("region", "level"): "REGION_IN_LEVEL",
            ("portal", "region"): "PORTAL_INCIDENT_TO_REGION",
        }[signature]
    return f"{relation.upper()}_{signature[0].upper()}_{signature[1].upper()}"


def _canonical_candidates(
    commitment: NominalCommitment,
    vocabulary: ASGVocabulary,
    possible_kinds: Iterable[ASGKind],
) -> list[tuple[ASGKind, str]]:
    candidates: list[tuple[ASGKind, str]] = []
    for kind in sorted(possible_kinds):
        if ("any" in _labels_for_kind(vocabulary, kind)
                and _canonical_form(commitment.head_text) in GENERIC_KIND_HEADS[kind]):
            candidates.append((kind, "any"))
        labels = [
            label
            for label in _labels_for_kind(vocabulary, kind)
            if label != "any"
        ]
        candidates.extend(
            (kind, label)
            for label in labels
            if _canonical_form(label) == _canonical_form(commitment.head_text)
        )
    return candidates


def _canonical_binding(
    commitment: NominalCommitment,
    vocabulary: ASGVocabulary,
    possible_kinds: Iterable[ASGKind],
) -> GroundingBinding | None:
    candidates = _canonical_candidates(commitment, vocabulary, possible_kinds)
    if len(candidates) != 1:
        return None
    kind, label = candidates[0]
    # A generic head proves the entity kind, not an unrestricted denotation
    # when its original nominal contains more information. For example,
    # head="floor", source="the upper floor" still needs label grounding.
    # Keep the enclosing clause as the LLM context, but check this nominal's
    # own span so modifiers on a neighbouring entity cannot affect the gate.
    if label == "any" and (
        _canonical_form(commitment.nominal_source_text or commitment.head_text)
        != _canonical_form(commitment.head_text)
    ):
        return None
    return GroundingBinding(
        head_text=commitment.head_text,
        asg_kind=kind,
        asg_label=label,
    )


def _canonical_form(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    value = value.replace("-", " ").replace("_", " ")
    words = _SPACE_RE.sub(" ", value).strip().split()
    while words and words[0] in _DETERMINERS:
        words.pop(0)
    if words:
        words[-1] = _singularize(words[-1])
    return " ".join(words)


def _visible_vocabulary(
    vocabulary: ASGVocabulary,
    items: tuple[GroundingItem, ...],
) -> ASGVocabulary:
    text = " ".join(f"{item.head_text} {item.source_text}" for item in items).casefold()
    level_relevant = any(word in text for word in _LEVEL_WORDS) or any(
        label != "any" and label.casefold() in text for label in vocabulary.level
    )
    if level_relevant:
        return vocabulary
    return vocabulary.model_copy(update={"level": []})


def _singularize(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("sses", "shes", "ches", "xes", "zes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _labels_for_kind(vocabulary: ASGVocabulary, kind: ASGKind) -> list[str]:
    return {
        "object": vocabulary.object,
        "region": vocabulary.region,
        "portal": vocabulary.portal,
        "level": vocabulary.level,
    }[kind]


def _positional_merge(
    occurrence_count: int,
    determined_positions: Iterable[int],
    determined: Iterable[GroundingBinding],
    residual_positions: Iterable[int],
    learned: Iterable[GroundingBinding],
) -> tuple[GroundingBinding, ...]:
    merged: list[GroundingBinding | None] = [None] * occurrence_count
    for position, binding in zip(determined_positions, determined):
        merged[position] = binding
    for position, binding in zip(residual_positions, learned):
        merged[position] = binding
    return tuple(binding for binding in merged if binding is not None)


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
