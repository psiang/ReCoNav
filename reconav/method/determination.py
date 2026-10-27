"""Shared determination vocabulary for language and world artifacts."""

from __future__ import annotations

import json
import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

FactOrigin = Literal["COMMIT", "PROPAGATION", "SOLVER"]
ActionEffect = Literal["CANONICALIZE", "RESTORE", "RESOLVE"]
ActionApplicationStatus = Literal["PENDING", "APPLIED", "STALE_PRECONDITION"]
Checkpoint = Literal[
    "contract_validation",
    "label_grounding",
    "scene_binding",
    "route_lowering",
    "route_verification",
]
DiagnosticSeverity = Literal["PASS", "INFO", "WARNING", "BLOCK"]
RuleCapability = Literal["DERIVE", "ASSERT", "REPAIR", "EVIDENCE"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Commitment(_StrictModel):
    key: str
    subject: str
    value: Any
    source: str
    evidence_refs: list[str] = Field(default_factory=list)


class Derivation(_StrictModel):
    rule: str
    premises: list[str] = Field(default_factory=list)


class DeterminedFact(_StrictModel):
    fact_id: str | None = None
    key: str
    subject: str
    value: Any
    origin: FactOrigin
    rule: str
    derivation: Derivation | None = None
    responsible_commitment: str | None = None
    entity_kind: str | None = None
    entity_id: str | None = None


class DeterministicAction(_StrictModel):
    checkpoint: Checkpoint
    effect: ActionEffect
    target: str
    before: Any
    after: Any
    rule: str
    supporting_facts: list[str] = Field(default_factory=list)
    invalidates: list[str] = Field(default_factory=list)
    target_key: str | None = None
    verified: bool = False
    application_status: ActionApplicationStatus = "PENDING"
    responsible_commitment: str | None = None
    entity_kind: str | None = None
    entity_id: str | None = None

    @property
    def rule_id(self) -> str:
        return self.rule

    @property
    def supporting_fact_ids(self) -> list[str]:
        return self.supporting_facts


class Diagnostic(_StrictModel):
    code: str
    severity: DiagnosticSeverity
    subject: str
    message: str
    assertion: str
    supporting_facts: list[str] = Field(default_factory=list)
    responsible_commitment: str | None = None
    entity_kind: str | None = None
    entity_id: str | None = None

    @property
    def rule_id(self) -> str:
        return self.assertion

    @property
    def supporting_fact_ids(self) -> list[str]:
        return self.supporting_facts


class Determination(_StrictModel):
    facts: list[DeterminedFact] = Field(default_factory=list)
    actions: list[DeterministicAction] = Field(default_factory=list)

    def value(self, key: str, default: Any = None) -> Any:
        for fact in reversed(self.facts):
            if fact.key == key:
                return fact.value
        return default

    def extended(
        self,
        *,
        facts: list[DeterminedFact] | None = None,
        actions: list[DeterministicAction] | None = None,
    ) -> Determination:
        return self.model_copy(
            update={
                "facts": [*self.facts, *(facts or [])],
                "actions": [*self.actions, *(actions or [])],
            }
        )


class CommitmentLedger:
    """Append-only commitments and their deterministic closure."""

    def __init__(self) -> None:
        self.commitments: list[Commitment] = []
        self.facts: list[DeterminedFact] = []
        self.actions: list[DeterministicAction] = []
        self.diagnostics: list[Diagnostic] = []

    def commit(self, commitment: Commitment) -> None:
        self.commitments.append(commitment)
        fact_id = _stable_fact_id(commitment.key, commitment.subject, commitment.value)
        self.facts.append(
            DeterminedFact(
                fact_id=fact_id,
                key=commitment.key,
                subject=commitment.subject,
                value=commitment.value,
                origin="COMMIT",
                rule=commitment.source,
                derivation=Derivation(
                    rule=commitment.source,
                    premises=list(commitment.evidence_refs),
                ),
                responsible_commitment=f"{commitment.subject}.{commitment.key}",
                entity_kind=commitment.subject.partition(":")[0] or None,
                entity_id=commitment.subject.partition(":")[2] or None,
            )
        )

    def fact(self, key: str, subject: str) -> DeterminedFact | None:
        for fact in reversed(self.facts):
            if fact.key == key and fact.subject == subject:
                return fact
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "commitments": [item.model_dump(mode="json") for item in self.commitments],
            "facts": [item.model_dump(mode="json") for item in self.facts],
            "actions": [item.model_dump(mode="json") for item in self.actions],
            "diagnostics": [item.model_dump(mode="json") for item in self.diagnostics],
        }


DeriveRule = Callable[[CommitmentLedger], Iterable[DeterminedFact]]
AssertRule = Callable[[CommitmentLedger], Iterable[Diagnostic]]
RepairRule = Callable[[CommitmentLedger], Iterable[DeterministicAction]]
EvidenceRule = Callable[[CommitmentLedger], Iterable[DeterminedFact]]
# Backwards-compatible type alias for the language-side determination engine.
Rule = DeriveRule


@dataclass(frozen=True)
class RegisteredRule:
    rule_id: str
    capabilities: frozenset[RuleCapability]
    derive_fn: DeriveRule | None = None
    assert_fn: AssertRule | None = None
    repair_fn: RepairRule | None = None
    evidence_fn: EvidenceRule | None = None
    learned_repair_diagnostic_codes: frozenset[str] = frozenset()
    learned_repair_task: str | None = None

    @property
    def derive(self) -> DeriveRule | None:
        """Compatibility accessor for pre-world-registry callers."""
        return self.derive_fn


class RuleRegistry:
    """Small registry for explicit Python rules shared across subsystems."""

    def __init__(self) -> None:
        self._rules: dict[str, RegisteredRule] = {}
        self._execution: dict[str, dict[str, Any]] = {}

    def register(
        self,
        rule_id: str,
        *,
        capabilities: Iterable[RuleCapability],
        derive: DeriveRule | None = None,
        derive_fn: DeriveRule | None = None,
        assert_fn: AssertRule | None = None,
        repair_fn: RepairRule | None = None,
        evidence_fn: EvidenceRule | None = None,
        learned_repair_diagnostic_codes: Iterable[str] = (),
        learned_repair_task: str | None = None,
    ) -> RegisteredRule:
        if not rule_id or "." not in rule_id:
            raise ValueError("rule_id must be a stable namespaced identifier")
        if derive is not None and derive_fn is not None:
            raise ValueError("provide derive or derive_fn, not both")
        callbacks = {
            "DERIVE": derive_fn or derive,
            "ASSERT": assert_fn,
            "REPAIR": repair_fn,
            "EVIDENCE": evidence_fn,
        }
        rule = RegisteredRule(
            rule_id,
            frozenset(capabilities),
            derive_fn=callbacks["DERIVE"],
            assert_fn=callbacks["ASSERT"],
            repair_fn=callbacks["REPAIR"],
            evidence_fn=callbacks["EVIDENCE"],
            learned_repair_diagnostic_codes=frozenset(learned_repair_diagnostic_codes),
            learned_repair_task=learned_repair_task,
        )
        for capability in rule.capabilities:
            if callbacks[capability] is None:
                raise ValueError(f"{capability.lower()} callback required for rule {rule_id}")
        if rule_id in self._rules:
            raise ValueError(f"duplicate rule_id: {rule_id}")
        if bool(rule.learned_repair_diagnostic_codes) != bool(rule.learned_repair_task):
            raise ValueError(
                "learned repair ownership requires both diagnostic codes and a task"
            )
        self._rules[rule_id] = rule
        self._execution[rule_id] = {
            "invocations": 0,
            "facts": 0,
            "diagnostics": 0,
            "actions": 0,
            "capability_invocations": {
                capability: 0 for capability in ("DERIVE", "ASSERT", "REPAIR", "EVIDENCE")
            },
            "capability_emissions": {
                capability: 0 for capability in ("DERIVE", "ASSERT", "REPAIR", "EVIDENCE")
            },
        }
        return rule

    def get(self, rule_id: str) -> RegisteredRule:
        return self._rules[rule_id]

    def rules(self, capability: RuleCapability | None = None) -> tuple[RegisteredRule, ...]:
        values = tuple(self._rules[key] for key in sorted(self._rules))
        if capability is None:
            return values
        return tuple(rule for rule in values if capability in rule.capabilities)

    def derive_rules(self) -> tuple[Rule, ...]:
        return tuple(rule.derive_fn for rule in self.rules("DERIVE") if rule.derive_fn is not None)

    def learned_repair_owners(self, diagnostic_code: str) -> tuple[RegisteredRule, ...]:
        """Return registered repair owners for an exact residual diagnostic code."""
        return tuple(
            rule
            for rule in self.rules()
            if diagnostic_code in rule.learned_repair_diagnostic_codes
        )

    def run_derivations_to_fixpoint(self, ledger: CommitmentLedger) -> list[DeterminedFact]:
        return DetermineEngine(self).determine_to_fixpoint(ledger)

    def run_assertions(self, ledger: CommitmentLedger) -> list[Diagnostic]:
        diagnostics = []
        for rule in self.rules("ASSERT"):
            self._execution[rule.rule_id]["invocations"] += 1
            self._execution[rule.rule_id]["capability_invocations"]["ASSERT"] += 1
            emitted = list(rule.assert_fn(ledger))  # type: ignore[misc]
            self._execution[rule.rule_id]["diagnostics"] += len(emitted)
            self._execution[rule.rule_id]["capability_emissions"]["ASSERT"] += len(emitted)
            diagnostics.extend(emitted)
        self._validate_emitted_rule_ids(diagnostics, kind="diagnostic")
        ledger.diagnostics.extend(diagnostics)
        return diagnostics

    def run_deterministic_repairs(self, ledger: CommitmentLedger) -> list[DeterministicAction]:
        actions = []
        for rule in self.rules("REPAIR"):
            self._execution[rule.rule_id]["invocations"] += 1
            self._execution[rule.rule_id]["capability_invocations"]["REPAIR"] += 1
            emitted = list(rule.repair_fn(ledger))  # type: ignore[misc]
            self._execution[rule.rule_id]["actions"] += len(emitted)
            self._execution[rule.rule_id]["capability_emissions"]["REPAIR"] += len(emitted)
            actions.extend(emitted)
        self._validate_emitted_rule_ids(actions, kind="action")
        if not actions:
            return []
        applied_actions = [self._apply_action(ledger, action) for action in actions]
        ledger.actions.extend(applied_actions)
        # Repairs change canonical commitments. All dependent closure/evidence
        # is invalidated and replayed so callers observe a new deterministic
        # frontier rather than a decorative action log.
        if any(action.application_status == "APPLIED" for action in applied_actions):
            ledger.facts = [fact for fact in ledger.facts if fact.origin == "COMMIT"]
            ledger.diagnostics.clear()
            self.run_derivations_to_fixpoint(ledger)
        return applied_actions

    def collect_evidence(self, ledger: CommitmentLedger) -> list[DeterminedFact]:
        collected = []
        for rule in self.rules("EVIDENCE"):
            self._execution[rule.rule_id]["invocations"] += 1
            self._execution[rule.rule_id]["capability_invocations"]["EVIDENCE"] += 1
            emitted = list(rule.evidence_fn(ledger))  # type: ignore[misc]
            self._execution[rule.rule_id]["facts"] += len(emitted)
            self._execution[rule.rule_id]["capability_emissions"]["EVIDENCE"] += len(emitted)
            collected.extend(emitted)
        facts = [
            fact if fact.fact_id is not None else fact.model_copy(
                update={"fact_id": _stable_fact_id(fact.key, fact.subject, fact.value)}
            )
            for fact in collected
        ]
        known = {_fact_identity(fact) for fact in ledger.facts}
        added = [fact for fact in facts if _fact_identity(fact) not in known]
        ledger.facts.extend(added)
        return added

    def _validate_emitted_rule_ids(
        self,
        values: Iterable[Diagnostic | DeterministicAction],
        *,
        kind: str,
    ) -> None:
        for value in values:
            rule_id = value.assertion if isinstance(value, Diagnostic) else value.rule
            if rule_id not in self._rules:
                raise ValueError(f"{kind} references unregistered rule_id: {rule_id}")
            capability = "ASSERT" if isinstance(value, Diagnostic) else "REPAIR"
            if capability not in self._rules[rule_id].capabilities:
                raise ValueError(
                    f"{kind} rule_id {rule_id} lacks required {capability} capability"
                )

    def _apply_action(
        self, ledger: CommitmentLedger, action: DeterministicAction
    ) -> DeterministicAction:
        """Apply an action to canonical commitments before replaying closure."""
        for index in range(len(ledger.commitments) - 1, -1, -1):
            old = ledger.commitments[index]
            if (
                old.subject == action.target
                and (action.target_key is None or old.key == action.target_key)
                and old.value == action.before
            ):
                replacement = old.model_copy(
                    update={"value": action.after, "source": action.rule}
                )
                ledger.commitments.pop(index)
                for fact_index in range(len(ledger.facts) - 1, -1, -1):
                    fact = ledger.facts[fact_index]
                    if fact.origin == "COMMIT" and fact.subject == old.subject and fact.key == old.key:
                        ledger.facts.pop(fact_index)
                        break
                ledger.commit(replacement)
                return action.model_copy(
                    update={"verified": True, "application_status": "APPLIED"}
                )
        # Never invent a solver-owned state when the advertised precondition did
        # not match.  The attempt remains auditable but is not an applied repair.
        return action.model_copy(
            update={"verified": False, "application_status": "STALE_PRECONDITION"}
        )

    def audit(self) -> list[dict[str, Any]]:
        result = []
        for rule in self.rules():
            callbacks = {
                capability: getattr(rule, f"{capability.lower()}_fn").__name__
                for capability in sorted(rule.capabilities)
            }
            result.append({
                "rule_id": rule.rule_id,
                "capabilities": sorted(rule.capabilities),
                "callbacks": callbacks,
                "learned_repair_diagnostic_codes": sorted(
                    rule.learned_repair_diagnostic_codes
                ),
                "learned_repair_task": rule.learned_repair_task,
                **self._execution[rule.rule_id],
            })
        return result


class DetermineEngine:
    """Run explicit typed Python rules until no new fact is produced."""

    def __init__(self, rules: Iterable[Rule] | RuleRegistry) -> None:
        self._registry = rules if isinstance(rules, RuleRegistry) else None
        self._rules = rules.derive_rules() if isinstance(rules, RuleRegistry) else tuple(rules)

    def determine_to_fixpoint(self, ledger: CommitmentLedger) -> list[DeterminedFact]:
        added: list[DeterminedFact] = []
        known = {_fact_identity(fact) for fact in ledger.facts}
        while True:
            round_added = []
            for rule in self._rules:
                owner = None
                if self._registry is not None:
                    owner = next((item for item in self._registry.rules("DERIVE") if item.derive_fn is rule), None)
                    if owner is not None:
                        self._registry._execution[owner.rule_id]["invocations"] += 1
                        self._registry._execution[owner.rule_id]["capability_invocations"]["DERIVE"] += 1
                for fact in rule(ledger):
                    if fact.fact_id is None:
                        fact = fact.model_copy(update={"fact_id": _stable_fact_id(fact.key, fact.subject, fact.value)})
                    identity = _fact_identity(fact)
                    if identity in known:
                        continue
                    known.add(identity)
                    ledger.facts.append(fact)
                    round_added.append(fact)
                    if owner is not None:
                        self._registry._execution[owner.rule_id]["facts"] += 1
                        self._registry._execution[owner.rule_id]["capability_emissions"]["DERIVE"] += 1
            if not round_added:
                return added
            added.extend(round_added)


def _fact_identity(fact: DeterminedFact) -> tuple[str, str, str]:
    return (
        fact.key,
        fact.subject,
        json.dumps(fact.value, sort_keys=True, separators=(",", ":"), default=str),
    )


def _stable_fact_id(key: str, subject: str, value: Any) -> str:
    payload = json.dumps([key, subject, value], sort_keys=True, separators=(",", ":"), default=str)
    return "fact:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
