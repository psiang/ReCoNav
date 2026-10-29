"""Paper methods and paired grounding/assurance ablations."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import time
from typing import Any

from reconav.solver.execution_profiles import ExecutionProfile
from reconav.method.verification.semantic_contract import audit_semantic_contract
from reconav.method.verification.verify import VerificationOutcome, run_verification
from reconav.method.grounding.label_grounding import (
    DetermineFirstSemanticGrounder,
    GroundingConfig,
    vocabulary_from_entity_payload,
)
from reconav.method.lowering.dsl import lower_grounded_contract, stage1_specification
from reconav.method.repair.semantic_repair import run_semantic_repair
from reconav.method.semantic.compiler import (
    SceneIndependentSemanticCompiler,
    SemanticCompilationResult,
)
from reconav.method.semantic.contracts import SceneIndependentSemanticContract
from reconav.method.runtime.backend import BoundedBackend, QueryBudgetExhausted
from reconav.method.runtime.accounting import complete_accounting
from reconav.method.runtime.binding import (
    BindingResolutionError,
    ForcedEarlyResolver,
    PinnedIdentityResolver,
    ProgressiveResolver,
    ReplayedFrontierResolver,
    label_grounding_artifact,
    learned_scene,
    seeded_random_identity_choice,
)
from reconav.method.runtime.model_calls import (
    ContextBudgetError,
    InputBudgetError,
    InputCharacterGuardError,
    RecordedClient,
)
from reconav.method.runtime.models import (
    METHODS,
    DirectAttemptResult,
    DirectDecision,
    NavigationRequest,
    RuntimeConfig,
    RUNTIME_VERSION,
    DirectIDSequenceDecision,
    method_name,
    method_backbone,
    normalize_method,
    method_provider,
    method_variant,
    normalize_provider,
    response_format,
)
from reconav.method.runtime.prompts import (
    direct_feedback_prompt,
    direct_prompt,
    prompt_payload,
    direct_id_sequence_prompt,
)
from reconav.method.runtime.scene import SceneContext


_DIRECT_VARIANTS = {"Direct", "Direct-Feedback"}
_SEMANTIC_VARIANTS = {
    "ReCoNav",
    "Forced-Early",
    "No-Verify-Repair",
    "Before-Repair",
}


class NavigationPipeline:
    def __init__(
        self,
        client,
        config: RuntimeConfig | None = None,
        *,
        backend_factory=None,
    ):
        self.client = client
        self.config = config or RuntimeConfig()
        self.backend_factory = backend_factory or (
            lambda: BoundedBackend(
                per_query_seconds=self.config.solver_seconds,
                total_seconds=self.config.resolution_seconds,
            )
        )

    def compile_semantics(self, request: NavigationRequest):
        recorder = self._recorder()
        compiler = SceneIndependentSemanticCompiler(
            client=recorder, **self.config.generation_kwargs()
        )
        try:
            with recorder.at("semantic_compilation"):
                result = compiler.compile(request.instruction)
        except Exception as exc:
            result = SemanticCompilationResult(
                "",
                None,
                None,
                infrastructure_error=f"{type(exc).__name__}: {exc}",
            )
        return result, recorder.calls

    def compile_label_grounding(
        self,
        request: NavigationRequest,
        contract,
        scene: SceneContext,
        *,
        grounding_policy: str = "determine_first",
    ):
        """Produce one validated label-grounding artifact.

        Determine-first is shared by No-Verify-Repair/Before-Repair/ReCoNav/Forced-Early.
        """
        recorder = self._recorder()
        if contract is None:
            return None, recorder.calls
        assurance = audit_semantic_contract(request.instruction, contract)
        if not assurance.accepted:
            return None, recorder.calls
        if grounding_policy != "determine_first":
            raise ValueError(f"unknown grounding policy {grounding_policy}")
        grounder_cls = DetermineFirstSemanticGrounder
        phase = "residual_label_grounding"
        try:
            grounder = grounder_cls(
                recorder,
                GroundingConfig(**self.config.generation_kwargs()),
            )
            with recorder.at(phase):
                run = grounder.ground(
                    contract,
                    vocabulary_from_entity_payload(scene.asg.to_entity_dict()),
                )
        except Exception as exc:
            failure = self._exception_failure_record(exc)
            artifact = label_grounding_artifact(
                contract,
                (),
                validation_success=False,
                diagnostics=(failure["reason"],),
            )
            artifact.update(
                raw_response=None,
                residual_llm_calls=0,
                expected_binding_count=artifact["occurrence_count"],
                response_schema_sha256=None,
                grounding_policy=grounding_policy,
                upstream_failure=failure,
            )
            return artifact, recorder.calls
        artifact = label_grounding_artifact(
            contract,
            run.final_bindings,
            validation_success=run.validation_success,
            diagnostics=run.validation_diagnostics,
            determination=run.determination,
        )
        artifact.update(
            raw_response=run.raw_response,
            residual_llm_calls=run.llm_calls,
            expected_binding_count=run.expected_binding_count,
            response_schema_sha256=run.response_schema_sha256,
            grounding_policy=grounding_policy,
        )
        return artifact, recorder.calls

    def run_progressive_frontiers(
        self,
        request: NavigationRequest,
        scene: SceneContext,
        *,
        semantic_result: SemanticCompilationResult,
        semantic_calls=None,
        semantic_reused: bool = False,
        label_grounding_result=None,
        label_grounding_calls=None,
        label_grounding_reused: bool = False,
        provider: str = "qwen",
        backbone: str | None = None,
        max_stage: int = 4,
    ) -> dict[str, dict[str, Any]]:
        """Advance one progressive frontier and freeze No-Verify-Repair/Before-Repair/ReCoNav snapshots.

        No-Verify-Repair, Before-Repair and ReCoNav are not three independent solver runs. One backend
        budget and one mutable frontier are advanced in order:
        semantic/label grounding -> candidate resolution -> verification -> repair.
        """
        if max_stage not in {2, 3, 4}:
            raise ValueError("progressive max_stage must be 2, 3, or 4")
        if request.scene_id != scene.asg.scene_id:
            raise ValueError("runtime request and ASG scene disagree")

        provider = normalize_provider(provider)
        backbone = backbone or provider
        if method_provider(method_name(backbone, "ReCoNav")) != provider:
            raise ValueError("backbone and transport provider disagree")
        client_provider = getattr(self.client, "provider", None)
        if (
            client_provider is not None
            and normalize_provider(client_provider) != provider
        ):
            raise ValueError(
                f"progressive frontier requires provider {provider}, "
                f"but pipeline client uses {client_provider}"
            )
        started = time.perf_counter()
        recorder = self._recorder()
        backend = self.backend_factory()
        prediction = self._base_prediction(
            request, scene, method_name(backbone, "No-Verify-Repair")
        )
        prediction["semantic_source"] = {
            "raw_response": semantic_result.raw_response,
            "candidate": semantic_result.candidate,
            "parse_error": semantic_result.parse_error,
            "infrastructure_error": semantic_result.infrastructure_error,
        }
        prediction["upstream"] = {
            "semantic_reused": semantic_reused,
            "semantic_calls": deepcopy(semantic_calls or []),
            "semantic_timing": semantic_result.timing,
            "label_grounding_reused": label_grounding_reused,
            "label_grounding_calls": deepcopy(label_grounding_calls or []),
        }
        contract = semantic_result.parsed_output
        if contract is not None:
            prediction["semantic_contract"] = contract.model_dump(mode="json")

        label_grounding_upstream_failure = (
            label_grounding_result.get("upstream_failure")
            if isinstance(label_grounding_result, dict)
            else None
        )
        if isinstance(label_grounding_upstream_failure, dict):
            prediction["label_grounding"] = deepcopy(label_grounding_result)
            self._failure(
                prediction,
                str(label_grounding_upstream_failure.get("stage", "model_response")),
                str(
                    label_grounding_upstream_failure.get(
                        "reason", "UPSTREAM_INFRASTRUCTURE_FAILURE"
                    )
                ),
                label_grounding_upstream_failure.get("detail"),
            )
        else:
            self._resolve_without_verification(
                request,
                contract,
                prediction,
                scene,
                recorder,
                backend,
                label_grounding_result=label_grounding_result,
                grounding_policy="determine_first",
            )

        charge_method = method_name(
            backbone,
            "ReCoNav" if max_stage == 4 else "Before-Repair" if max_stage == 3 else "No-Verify-Repair",
        )
        snapshot_fallback_elapsed = 0.0

        def freeze(method: str, artifact: dict[str, Any]) -> dict[str, Any]:
            nonlocal snapshot_fallback_elapsed
            frozen = deepcopy(artifact)
            frozen["method"] = method
            frozen_variant = method_variant(method)
            frozen["method_variant"] = frozen_variant
            frozen["llm_provider"] = method_provider(method)

            fallback_recorder = self._recorder()
            fallback_backend = backend.fork_remaining()
            fallback_seconds = 0.0
            if self._final_candidate_choice_needed(frozen):
                fallback_started = time.perf_counter()
                self._apply_final_identity_fallback(
                    request=request,
                    scene=scene,
                    prediction=frozen,
                    recorder=fallback_recorder,
                    backend=fallback_backend,
                )
                fallback_seconds = time.perf_counter() - fallback_started
                snapshot_fallback_elapsed += fallback_seconds

            self._refresh_identity_commitment(frozen)
            frozen["progressive_frontier"] = {
                "single_run": True,
                "stage": (
                    "No-Verify-Repair"
                    if frozen_variant == "No-Verify-Repair"
                    else "Before-Repair"
                    if frozen_variant == "Before-Repair"
                    else "ReCoNav"
                ),
                "sequence": ["No-Verify-Repair", "Before-Repair", "ReCoNav"],
            }
            combined_calls = [
                *deepcopy(recorder.calls),
                *deepcopy(fallback_recorder.calls),
            ]
            combined_queries = [
                *deepcopy(backend.queries),
                *deepcopy(fallback_backend.queries),
            ]
            frozen["model_calls"] = combined_calls
            frozen["solver_queries"] = combined_queries
            frozen.setdefault("timing", {})
            shared_elapsed = (
                time.perf_counter()
                - started
                - snapshot_fallback_elapsed
            )
            frozen["timing"]["fallback_pipeline_seconds"] = fallback_seconds
            frozen["timing"]["own_pipeline_seconds"] = (
                shared_elapsed + fallback_seconds
            )
            frozen["timing"]["shared_direct_first_pipeline_seconds"] = 0.0
            frozen["timing"]["shared_progressive_binding_pipeline_seconds"] = 0.0
            frozen["timing"]["pipeline_seconds"] = frozen["timing"][
                "own_pipeline_seconds"
            ]
            frozen["timing"]["solver_seconds"] = (
                backend.elapsed + fallback_backend.elapsed
            )

            own = sum(
                bool(call.get("provider_called"))
                for call in combined_calls
            )
            semantic_actual = sum(
                bool(call.get("provider_called"))
                for call in semantic_calls or []
            )
            label_grounding_actual = sum(
                bool(call.get("provider_called"))
                for call in label_grounding_calls or []
            )
            charge_upstream = method == charge_method
            frozen["call_accounting"] = {
                "own_model_calls": own,
                "semantic_calls_required": semantic_actual,
                "semantic_calls_charged_this_run": (
                    semantic_actual
                    if charge_upstream and not semantic_reused
                    else 0
                ),
                "label_grounding_calls_required": label_grounding_actual,
                "label_grounding_calls_charged_this_run": (
                    label_grounding_actual
                    if charge_upstream and not label_grounding_reused
                    else 0
                ),
                "direct_first_calls_required": 0,
                "direct_first_calls_charged_this_run": 0,
                "total_logical_calls_including_reused_upstream": (
                    own + semantic_actual + label_grounding_actual
                ),
            }
            return complete_accounting(frozen)

        no_verify_repair_method = method_name(backbone, "No-Verify-Repair")
        before_repair_method = method_name(backbone, "Before-Repair")
        reconav_method = method_name(backbone, "ReCoNav")
        outputs: dict[str, dict[str, Any]] = {
            no_verify_repair_method: freeze(no_verify_repair_method, prediction)
        }
        if isinstance(label_grounding_upstream_failure, dict):
            if max_stage >= 3:
                blocked_before_repair = deepcopy(prediction)
                blocked_before_repair["assurance"] = {
                    "not_run_reason": "shared_label_grounding_upstream_failure",
                    "repair": None,
                }
                outputs[before_repair_method] = freeze(
                    before_repair_method, blocked_before_repair
                )
            if max_stage >= 4:
                blocked_ours = deepcopy(prediction)
                blocked_ours["assurance"] = {
                    "not_run_reason": "shared_label_grounding_upstream_failure",
                    "repair": None,
                }
                outputs[reconav_method] = freeze(
                    reconav_method, blocked_ours
                )
            return outputs
        if max_stage == 2:
            return outputs

        shared_history = deepcopy(
            prediction.get("candidate_resolution_history", [])
        )
        replay_resolver = ReplayedFrontierResolver(
            prediction.get("candidate_resolution"),
            ProgressiveResolver(
                backend, max_assignments=self.config.max_assignments
            ),
        )
        outcome = run_verification(
            instruction=request.instruction,
            robot_position=request.robot_position,
            artifact=prediction,
            asg=scene.asg,
            force_recompute=False,
            binding_resolver=replay_resolver,
            execution_backend=backend,
        )
        before_repair = deepcopy(outcome.prediction_artifact)
        before_repair["assurance"] = {
            "final": self._outcome_summary(outcome),
            "repair": None,
        }
        before_repair["candidate_resolution_history"] = [
            *shared_history,
            *deepcopy(replay_resolver.history),
        ]
        before_repair["method_policy"].update(
            {
                "grounding_policy": "determine_first",
                "candidate_policy": "progressive_constraint_resolution",
                "verification_enabled": True,
                "repair_enabled": False,
                "commitment_rule": (
                    "local determination; joint hard feasibility; explicit preference; "
                    "otherwise unresolved"
                ),
            }
        )
        outputs[before_repair_method] = freeze(before_repair_method, before_repair)
        if max_stage == 3:
            return outputs

        # ReCoNav starts from this exact Before-Repair outcome. It never re-runs verification.
        before = self._outcome_summary(outcome)
        repair = None
        repair_resolver = ProgressiveResolver(
            backend, max_assignments=self.config.max_assignments
        )
        if (
            self.config.repair_rounds
            and not outcome.verification_accepted
            and not outcome.needs_backend_fix
        ):
            repair = run_semantic_repair(
                instruction=request.instruction,
                robot_position=request.robot_position,
                starting_outcome=outcome,
                asg=scene.asg,
                call_model=self._repair_call(recorder),
                max_rounds=self.config.repair_rounds,
                include_warnings=False,
                binding_resolver=repair_resolver,
                execution_backend=backend,
                preflight=self._repair_preflight(recorder),
            )
            outcome = repair.final_outcome

        reconav = deepcopy(outcome.prediction_artifact)
        reconav["assurance"] = {
            "before_repair": before,
            "final": self._outcome_summary(outcome),
            "repair": (
                None
                if repair is None
                else {
                    "rounds": list(repair.rounds),
                    "stopped_reason": repair.stopped_reason,
                    "model_calls": repair.model_calls,
                }
            ),
        }
        reconav["candidate_resolution_history"] = [
            *shared_history,
            *deepcopy(replay_resolver.history),
            *deepcopy(repair_resolver.history),
        ]
        reconav["method_policy"].update(
            {
                "grounding_policy": "determine_first",
                "candidate_policy": "progressive_constraint_resolution",
                "verification_enabled": True,
                "repair_enabled": True,
                "commitment_rule": (
                    "local determination; joint hard feasibility; explicit preference; "
                    "otherwise unresolved until the common uncertified execution fallback"
                ),
                "final_execution_fallback": True,
            }
        )
        outputs[reconav_method] = freeze(reconav_method, reconav)
        return outputs

    def compile_direct(self, request: NavigationRequest, scene: SceneContext):
        """Produce the shared first Direct attempt used by Direct and Direct+Feedback."""
        recorder = self._recorder()
        started = time.perf_counter()
        try:
            with recorder.at("direct_semantic_and_instance_commitment"):
                raw = recorder.generate(
                    direct_prompt(),
                    prompt_payload(
                        request,
                        scene.model_view(self.config.scene_geometry),
                    ),
                    response_format=response_format(
                        DirectDecision, "direct_navigation_contract"
                    ),
                    **self.config.generation_kwargs(),
                )
        except Exception as exc:
            return (
                DirectAttemptResult(
                    "",
                    None,
                    infrastructure_error=f"{type(exc).__name__}: {exc}",
                    timing={"total_seconds": time.perf_counter() - started},
                ),
                recorder.calls,
            )
        try:
            parsed = DirectDecision.model_validate_json(raw)
        except Exception as exc:
            result = DirectAttemptResult(
                raw,
                None,
                parse_error=f"{type(exc).__name__}: {exc}",
                timing={"total_seconds": time.perf_counter() - started},
            )
        else:
            result = DirectAttemptResult(
                raw,
                parsed,
                timing={"total_seconds": time.perf_counter() - started},
            )
        return result, recorder.calls

    def run(
        self,
        request: NavigationRequest,
        scene: SceneContext,
        method: str,
        *,
        semantic_result: SemanticCompilationResult | None = None,
        semantic_calls=None,
        semantic_reused: bool = False,
        label_grounding_result=None,
        label_grounding_calls=None,
        label_grounding_reused: bool = False,
        direct_result: DirectAttemptResult | None = None,
        direct_calls=None,
        direct_reused: bool = False,
        direct_first_prediction: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        method = normalize_method(method)
        if method not in METHODS:
            raise ValueError(f"unknown method {method}")
        provider = method_provider(method)
        variant = method_variant(method)
        client_provider = getattr(self.client, "provider", None)
        if client_provider is not None and normalize_provider(client_provider) != provider:
            raise ValueError(
                f"method {method} requires provider {provider}, "
                f"but pipeline client uses {client_provider}"
            )
        if request.scene_id != scene.asg.scene_id:
            raise ValueError("runtime request and ASG scene disagree")
        if variant in {"No-Verify-Repair", "Before-Repair", "ReCoNav"}:
            if semantic_result is None:
                semantic_result, semantic_calls = self.compile_semantics(request)
            if label_grounding_result is None:
                contract = semantic_result.parsed_output
                label_grounding_result, label_grounding_calls = self.compile_label_grounding(
                    request,
                    contract,
                    scene,
                    grounding_policy="determine_first",
                )
            max_stage = (
                4
                if variant == "ReCoNav"
                else 3
                if variant == "Before-Repair"
                else 2
            )
            return self.run_progressive_frontiers(
                request,
                scene,
                semantic_result=semantic_result,
                semantic_calls=semantic_calls,
                semantic_reused=semantic_reused,
                label_grounding_result=label_grounding_result,
                label_grounding_calls=label_grounding_calls,
                label_grounding_reused=label_grounding_reused,
                provider=provider,
                backbone=method_backbone(method),
                max_stage=max_stage,
            )[method]

        started = time.perf_counter()
        recorder = self._recorder()
        backend = self.backend_factory()
        prediction = self._base_prediction(request, scene, method)
        inline_upstream_seconds = 0.0

        try:
            if variant in _DIRECT_VARIANTS:
                if direct_result is None:
                    direct_result, direct_calls = self.compile_direct(request, scene)
                    inline_upstream_seconds += float((direct_result.timing or {}).get("total_seconds", 0.0))
                prediction["direct_source"] = {
                    "raw_response": direct_result.raw_response,
                    "parse_error": direct_result.parse_error,
                    "infrastructure_error": direct_result.infrastructure_error,
                    "timing": direct_result.timing,
                    "reused": direct_reused,
                    "model_calls": deepcopy(direct_calls or []),
                }
                if variant == "Direct":
                    self._apply_direct_attempt(
                        request,
                        scene,
                        direct_result,
                        backend,
                        prediction,
                    )
                else:
                    self._run_direct_feedback(
                        request,
                        scene,
                        direct_result,
                        backend,
                        prediction,
                        recorder,
                        feedback_method=method,
                        first_prediction=direct_first_prediction,
                    )
            elif variant == "Direct-ID-Sequence":
                self._run_direct_id_sequence(
                    request, scene, backend, prediction, recorder
                )
            else:
                if semantic_result is None:
                    semantic_result, semantic_calls = self.compile_semantics(request)
                    inline_upstream_seconds += float((semantic_result.timing or {}).get("total_seconds", 0.0))
                prediction["semantic_source"] = {
                    "raw_response": semantic_result.raw_response,
                    "candidate": semantic_result.candidate,
                    "parse_error": semantic_result.parse_error,
                    "infrastructure_error": semantic_result.infrastructure_error,
                }
                prediction["upstream"] = {
                    "semantic_reused": semantic_reused,
                    "semantic_calls": deepcopy(semantic_calls or []),
                    "semantic_timing": semantic_result.timing,
                    "label_grounding_reused": label_grounding_reused,
                    "label_grounding_calls": deepcopy(label_grounding_calls or []),
                }
                contract = semantic_result.parsed_output
                if contract is not None:
                    prediction["semantic_contract"] = contract.model_dump(mode="json")
                label_grounding_upstream_failure = (
                    label_grounding_result.get("upstream_failure")
                    if isinstance(label_grounding_result, dict)
                    else None
                )
                if isinstance(label_grounding_upstream_failure, dict):
                    prediction["label_grounding"] = deepcopy(label_grounding_result)
                    self._failure(
                        prediction,
                        str(label_grounding_upstream_failure.get("stage", "model_response")),
                        str(
                            label_grounding_upstream_failure.get(
                                "reason", "UPSTREAM_INFRASTRUCTURE_FAILURE"
                            )
                        ),
                        label_grounding_upstream_failure.get("detail"),
                    )
                elif variant == "Forced-Early":
                    self._forced_early(
                        request,
                        contract,
                        prediction,
                        scene,
                        recorder,
                        backend,
                        label_grounding_result=label_grounding_result,
                    )
                else:  # pragma: no cover - protected by METHODS
                    raise ValueError(f"unsupported method {method}")
        except Exception as exc:
            self._classify_unhandled_failure(prediction, exc)
        finally:
            own_pipeline_seconds = time.perf_counter() - started
            shared_first_solver_queries = []
            shared_first_pipeline_seconds = 0.0
            shared_first_solver_seconds = 0.0
            if (
                variant == "Direct-Feedback"
                and isinstance(direct_first_prediction, dict)
            ):
                shared_first_solver_queries = deepcopy(
                    direct_first_prediction.get("solver_queries", [])
                )
                first_timing = direct_first_prediction.get("timing")
                if isinstance(first_timing, dict):
                    shared_first_pipeline_seconds = float(
                        first_timing.get("execution_pipeline_seconds", first_timing.get("pipeline_seconds", 0.0))
                    )
                    shared_first_solver_seconds = float(
                        first_timing.get("solver_seconds", 0.0)
                    )

            prediction["model_calls"] = recorder.calls
            prediction["solver_queries"] = [
                *shared_first_solver_queries,
                *deepcopy(backend.queries),
            ]
            prediction.setdefault("timing", {})
            prediction["timing"]["own_pipeline_seconds"] = own_pipeline_seconds
            prediction["timing"]["shared_direct_first_pipeline_seconds"] = (
                shared_first_pipeline_seconds
            )
            prediction["timing"]["shared_progressive_binding_pipeline_seconds"] = 0.0
            prediction["timing"]["pipeline_seconds"] = (
                own_pipeline_seconds + shared_first_pipeline_seconds
            )
            prediction["timing"]["solver_seconds"] = (
                backend.elapsed + shared_first_solver_seconds
            )

            own = sum(bool(call.get("provider_called")) for call in recorder.calls)
            semantic_actual = sum(
                bool(call.get("provider_called")) for call in semantic_calls or []
            )
            label_grounding_actual = sum(
                bool(call.get("provider_called")) for call in label_grounding_calls or []
            )
            direct_actual = sum(
                bool(call.get("provider_called")) for call in direct_calls or []
            )
            uses_semantic = variant in _SEMANTIC_VARIANTS
            uses_label_grounding = variant in _SEMANTIC_VARIANTS
            uses_direct = variant in _DIRECT_VARIANTS
            prediction["call_accounting"] = {
                "own_model_calls": own,
                "semantic_calls_required": semantic_actual if uses_semantic else 0,
                "semantic_calls_charged_this_run": (
                    0 if semantic_reused or not uses_semantic else semantic_actual
                ),
                "label_grounding_calls_required": label_grounding_actual if uses_label_grounding else 0,
                "label_grounding_calls_charged_this_run": (
                    0 if label_grounding_reused or not uses_label_grounding else label_grounding_actual
                ),
                "direct_first_calls_required": direct_actual if uses_direct else 0,
                "direct_first_calls_charged_this_run": (
                    0 if direct_reused or not uses_direct else direct_actual
                ),
                "total_logical_calls_including_reused_upstream": (
                    own
                    + (semantic_actual if uses_semantic else 0)
                    + (label_grounding_actual if uses_label_grounding else 0)
                    + (direct_actual if uses_direct else 0)
                ),
            }
        return complete_accounting(prediction, inline_upstream_seconds=inline_upstream_seconds)

    def _base_prediction(
        self,
        request: NavigationRequest,
        scene: SceneContext,
        method: str,
    ) -> dict[str, Any]:
        return {
            "task_id": request.task_id,
            "scene_id": request.scene_id,
            "instruction": request.instruction,
            "robot_position": request.robot_position,
            "method": method,
            "method_variant": method_variant(method),
            "llm_provider": method_provider(method),
            "model_backbone": method_backbone(method),
            "runtime_version": RUNTIME_VERSION,
            "input_source": {
                "data_source": scene.data_source,
                "path": scene.source_path,
                "fingerprint": scene.fingerprint,
            },
            "semantic_contract": None,
            "predicted": {},
            "route": {},
            "failure": None,
            "scene_binding": {"executed": False, "success": False},
            "lowering": {"executed": False},
            "execution": {"executed": False, "solver_success": False},
            "stage_verification": {
                "json_valid": False,
                "schema_valid": False,
                "dsl_valid": False,
                "solver_success": False,
            },
            "method_policy": {
                "random_grounding_fallback": False,
                "semantic_retry_baselines": False,
                "baseline_solver_candidate_selection": False,
                "reconav_identity_length_tie_break": False,
            },
            "metric_applicability": {"semantic_contract": True},
            "timing": {},
        }

    def _apply_direct_attempt(
        self,
        request,
        scene,
        attempt: DirectAttemptResult,
        backend,
        prediction,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Validate/execute one Direct artifact; never repair it."""
        prediction["failure"] = None
        prediction["route"] = {}
        prediction["scene_binding"] = {"executed": False, "success": False}
        prediction["lowering"] = {"executed": False}
        prediction["execution"] = {"executed": False, "solver_success": False}
        prediction["predicted"] = {}
        if attempt.infrastructure_error:
            self._failure(
                prediction,
                "model_response",
                "UPSTREAM_INFRASTRUCTURE_FAILURE",
                attempt.infrastructure_error,
            )
            return False, self._direct_feedback_record(prediction)
        if attempt.parsed_output is None:
            self._failure(
                prediction,
                "model_response",
                "PARSE_FAILED",
                attempt.parse_error or "Direct response did not validate",
            )
            return False, self._direct_feedback_record(prediction)

        parsed = attempt.parsed_output
        prediction["stage_verification"].update(
            json_valid=True, schema_valid=True
        )
        prediction["learned_decision"] = parsed.model_dump(mode="json")
        if parsed.semantic_contract is not None:
            prediction["semantic_contract"] = parsed.semantic_contract.model_dump(
                mode="json"
            )
            prediction["predicted"]["specification"] = stage1_specification(
                parsed.semantic_contract
            )
        if parsed.status == "unresolved" or parsed.semantic_contract is None:
            self._failure(
                prediction,
                "grounding",
                "MODEL_UNRESOLVED",
                parsed.reason,
            )
            return False, self._direct_feedback_record(prediction)

        contract = parsed.semantic_contract
        prediction["semantic_contract"] = contract.model_dump(mode="json")
        try:
            self._check_contract(request, contract, prediction)
        except Exception as exc:
            self._failure(
                prediction,
                "semantic",
                "INVALID_SEMANTIC_CONTRACT",
                f"{type(exc).__name__}: {exc}",
            )
            return False, self._direct_feedback_record(prediction)

        try:
            bound, label_grounding = learned_scene(
                contract,
                parsed.bindings,
                scene.asg,
                request.robot_position,
            )
        except Exception as exc:
            self._failure(
                prediction,
                "grounding",
                "INVALID_LEARNED_CONTRACT",
                f"{type(exc).__name__}: {exc}",
            )
            return False, self._direct_feedback_record(prediction)

        prediction["label_grounding"] = label_grounding
        try:
            self._execute_baseline(
                request, contract, bound, scene, backend, prediction
            )
        except QueryBudgetExhausted as exc:
            self._failure(
                prediction,
                "solver",
                "SOLVER_UNKNOWN",
                f"{type(exc).__name__}: {exc}",
            )
        except Exception as exc:
            stage = "solver" if prediction["lowering"].get("executed") else "lowering"
            self._failure(
                prediction,
                stage,
                "BACKEND_ERROR" if stage == "solver" else "LOWERING_ERROR",
                f"{type(exc).__name__}: {exc}",
            )
        return prediction.get("failure") is None, self._direct_feedback_record(
            prediction
        )

    def _apply_feedback_attempt(self, request, scene, attempt, backend, trial):
        """Keep candidate-validation failures inside their feedback round.

        Classify against the partially populated trial, as Direct's outer
        handler does. In particular, an ambiguous event during specification
        rendering is an invalid learned artifact, not a provider outage.
        Direct and the other methods retain their existing execution paths.
        """
        try:
            return self._apply_direct_attempt(request, scene, attempt, backend, trial)
        except Exception as exc:
            self._classify_unhandled_failure(trial, exc)
            return False, self._direct_feedback_record(trial)

    def _run_direct_feedback(
        self,
        request,
        scene,
        first_attempt,
        backend,
        prediction,
        recorder,
        *,
        feedback_method: str,
        first_prediction=None,
    ):
        """Run bounded Direct feedback from the exact completed Direct result.

        Attempt 1 is not revalidated or re-solved when a completed Direct
        prediction is supplied by the paired runner.  Each later attempt sees
        the same original task/ASG input plus the immediately previous complete
        model output and its mechanical validation/lowering/solver feedback,
        then must regenerate a full new contract+binding artifact.
        """
        base = deepcopy(prediction)
        history: list[dict[str, Any]] = []

        if isinstance(first_prediction, dict):
            first_trial = deepcopy(first_prediction)
            first_success = first_trial.get("failure") is None
            first_feedback = self._direct_feedback_record(first_trial)
        else:
            # Standalone fallback outside the paired experiment runner.
            first_trial = deepcopy(base)
            first_success, first_feedback = self._apply_feedback_attempt(
                request,
                scene,
                first_attempt,
                backend,
                first_trial,
            )

        history.append(
            {
                "attempt": 1,
                "is_shared_direct_first_attempt": True,
                "reused_complete_direct_result": isinstance(
                    first_prediction, dict
                ),
                "success": first_success,
                "feedback": first_feedback,
            }
        )

        if self._direct_infrastructure_failure(first_trial):
            prediction.clear()
            prediction.update(first_trial)
            prediction["method"] = feedback_method
            prediction["method_variant"] = method_variant(feedback_method)
            prediction["llm_provider"] = method_provider(feedback_method)
            if isinstance(prediction.get("direct_source"), dict):
                prediction["direct_source"]["reused"] = True
            prediction["direct_feedback"] = {
                "max_feedback_rounds": self.config.feedback_rounds,
                "attempts": history,
                "stopped_reason": "infrastructure_failure_not_repaired",
            }
            return

        if first_success or self.config.feedback_rounds == 0:
            prediction.clear()
            prediction.update(first_trial)
            prediction["method"] = feedback_method
            prediction["method_variant"] = method_variant(feedback_method)
            prediction["llm_provider"] = method_provider(feedback_method)
            if isinstance(prediction.get("direct_source"), dict):
                prediction["direct_source"]["reused"] = True
            prediction["direct_feedback"] = {
                "max_feedback_rounds": self.config.feedback_rounds,
                "attempts": history,
            }
            return

        attempt = first_attempt
        feedback = first_feedback
        for revision_index in range(1, self.config.feedback_rounds + 1):
            payload = {
                "instruction": request.instruction,
                "original_start": request.robot_position,
                "start_heading": request.heading,
                "asg": scene.model_view(self.config.scene_geometry),
                "previous_raw_response": attempt.raw_response,
                "mechanical_feedback": feedback,
            }
            started = time.perf_counter()
            try:
                with recorder.at("direct_feedback_revision"):
                    raw = recorder.generate(
                        direct_feedback_prompt(),
                        json.dumps(
                            payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        response_format=response_format(
                            DirectDecision,
                            "direct_feedback_contract",
                        ),
                        **self.config.generation_kwargs(),
                    )
            except Exception as exc:
                attempt = DirectAttemptResult(
                    "",
                    None,
                    infrastructure_error=f"{type(exc).__name__}: {exc}",
                    timing={
                        "total_seconds": time.perf_counter() - started
                    },
                )
            else:
                try:
                    parsed = DirectDecision.model_validate_json(raw)
                except Exception as exc:
                    attempt = DirectAttemptResult(
                        raw,
                        None,
                        parse_error=f"{type(exc).__name__}: {exc}",
                        timing={
                            "total_seconds": time.perf_counter() - started
                        },
                    )
                else:
                    attempt = DirectAttemptResult(
                        raw,
                        parsed,
                        timing={
                            "total_seconds": time.perf_counter() - started
                        },
                    )

            trial = deepcopy(base)
            success, feedback = self._apply_feedback_attempt(
                request,
                scene,
                attempt,
                backend,
                trial,
            )
            history.append(
                {
                    "attempt": revision_index + 1,
                    "is_shared_direct_first_attempt": False,
                    "reused_complete_direct_result": False,
                    "success": success,
                    "feedback": feedback,
                }
            )
            infrastructure_failure = self._direct_infrastructure_failure(trial)
            if (
                success
                or infrastructure_failure
                or revision_index >= self.config.feedback_rounds
            ):
                prediction.clear()
                prediction.update(trial)
                prediction["method"] = feedback_method
                prediction["method_variant"] = method_variant(feedback_method)
                prediction["llm_provider"] = method_provider(feedback_method)
                if isinstance(prediction.get("direct_source"), dict):
                    prediction["direct_source"]["reused"] = True
                prediction["direct_feedback"] = {
                    "max_feedback_rounds": self.config.feedback_rounds,
                    "attempts": history,
                    "stopped_reason": (
                        "infrastructure_failure_not_repaired"
                        if infrastructure_failure
                        else "completed"
                        if success
                        else "feedback_budget_exhausted"
                    ),
                }
                return

    @staticmethod
    def _direct_infrastructure_failure(prediction) -> bool:
        failure = prediction.get("failure")
        if not isinstance(failure, dict):
            return False
        return failure.get("reason") in {
            "UPSTREAM_INFRASTRUCTURE_FAILURE",
            "CONTEXT_BUDGET_EXCEEDED",
            "INPUT_CHARACTER_GUARD_EXCEEDED",
            "INPUT_BUDGET_EXCEEDED",
        }

    @staticmethod
    def _direct_feedback_record(prediction):
        failure = prediction.get("failure")
        route = prediction.get("route")
        return {
            "failure": deepcopy(failure),
            "solver_status": (
                route.get("status") if isinstance(route, dict) else None
            ),
            "solver_reason": (
                route.get("status_reason") if isinstance(route, dict) else None
            ),
        }

    def _run_direct_id_sequence(
        self,
        request,
        scene,
        backend,
        prediction,
        recorder,
    ):
        prediction["metric_applicability"]["semantic_contract"] = False
        generation = self.config.generation_kwargs()
        # Direct ID Sequence emits only a bounded typed-ID sequence. Greedy decoding plus
        # the minified-JSON prompt avoids sampling into vLLM's otherwise
        # unbounded structural-whitespace branch.
        generation["max_tokens"] = min(int(generation["max_tokens"]), 1024)
        generation["temperature"] = 0.0
        generation["top_p"] = 1.0
        with recorder.at("direct_id_sequence"):
            raw = recorder.generate(
                direct_id_sequence_prompt(),
                prompt_payload(
                    request,
                    scene.model_view(self.config.scene_geometry),
                ),
                response_format=response_format(
                    DirectIDSequenceDecision, "direct_id_sequence"
                ),
                **generation,
            )
        try:
            parsed = DirectIDSequenceDecision.model_validate_json(raw)
        except Exception as exc:
            stripped = raw.rstrip()
            reason = (
                "MODEL_OUTPUT_TRUNCATED"
                if stripped and not stripped.endswith("}")
                else "INVALID_MODEL_OUTPUT"
            )
            self._failure(
                prediction,
                "model_response",
                reason,
                f"{type(exc).__name__}: {exc}",
            )
            return
        prediction["native_output"] = parsed.model_dump(mode="json")
        prediction["stage_verification"].update(
            json_valid=True, schema_valid=True
        )
        if not parsed.targets:  # protected by the strict response schema
            self._failure(
                prediction,
                "planning",
                "INVALID_DIRECT_ID_SEQUENCE",
                "resolved ID sequence must contain at least one typed ASG ID",
            )
            return
        if parsed.targets[-1].kind == "portal":
            self._failure(
                prediction,
                "planning",
                "INVALID_DIRECT_ID_SEQUENCE",
                "final route target cannot be a portal",
            )
            return
        lookups = {
            "object": scene.asg.get_object,
            "region": scene.asg.get_region,
            "portal": scene.asg.get_portal,
            "level": scene.asg.get_level,
        }
        for target in parsed.targets:
            if lookups[target.kind](target.id) is None:
                self._failure(
                    prediction,
                    "planning",
                    "INVALID_DIRECT_ID_SEQUENCE",
                    f"missing {target.kind} ID {target.id}",
                )
                return

        start = request.robot_position
        lines = [
            f"START POS {start['x']:.12g} {start['floor_height']:.12g} {start['y']:.12g}"
        ]
        dsl_kind = {
            "object": "OBJ",
            "region": "REGION",
            "portal": "PORTAL",
            "level": "LEVEL",
        }
        for index, target in enumerate(parsed.targets[:-1], start=1):
            lines.append(
                f"v{index}: VIA {dsl_kind[target.kind]} {int(target.id)}"
            )
        goal = parsed.targets[-1]
        lines.append(f"GOAL {dsl_kind[goal.kind]} {int(goal.id)}")
        dsl = "\n".join(lines) + "\n"
        prediction["lowering"] = {
            "executed": True,
            "dsl_valid": True,
            "dsl": dsl,
            "origin": "direct_id_sequence_only",
        }
        route = backend(
            dsl,
            scene.asg,
            profile=ExecutionProfile.WAYPOINT,
        )
        prediction["route"] = route
        found = route.get("status") == "SAT" and bool(route.get("waypoints"))
        prediction["execution"] = {
            "executed": True,
            "solver_success": found,
            "status": route.get("status"),
            "profile": ExecutionProfile.WAYPOINT.value,
        }
        prediction["stage_verification"].update(
            dsl_valid=True, solver_success=found
        )
        if not found:
            status = route.get("status", "UNKNOWN")
            reason = (
                "EMPTY_ROUTE"
                if status == "SAT" and not route.get("waypoints")
                else status
            )
            self._failure(
                prediction,
                "route",
                reason,
                route.get("status_reason")
                or (
                    "WAYPOINT solver returned SAT without an executable waypoint witness"
                    if reason == "EMPTY_ROUTE"
                    else None
                ),
            )

    @staticmethod
    def _failure(prediction, stage, reason, detail):
        prediction["failure"] = {
            "stage": stage,
            "reason": reason,
            "detail": detail,
        }
        return prediction

    @staticmethod
    def _check_contract(request, contract, prediction):
        assurance = audit_semantic_contract(request.instruction, contract)
        prediction["semantic_assurance"] = assurance.model_dump(mode="json")
        prediction["stage_verification"].update(
            json_valid=True, schema_valid=True
        )
        if not assurance.accepted:
            raise ValueError(
                "semantic structure invalid: "
                + json.dumps(
                    [
                        diagnostic.model_dump(mode="json")
                        for diagnostic in assurance.diagnostics
                    ]
                )
            )
        prediction["predicted"]["specification"] = stage1_specification(contract)

    def _execute_baseline(
        self,
        request,
        contract,
        bound,
        scene,
        backend,
        prediction,
        *,
        origin: str = "learned_commitment",
    ):
        lowered = lower_grounded_contract(
            contract, bound, request.robot_position, scene.asg
        )
        prediction["scene_binding"] = {
            "executed": True,
            "success": True,
            "grounded_contract": bound.model_dump(mode="json"),
            "origin": origin,
        }
        report = prediction.get("candidate_resolution")
        mechanism = report.get("mechanism") if isinstance(report, dict) else None
        if origin == "fallback_random":
            commitment_source, certified = "fallback", False
        elif mechanism == "forced_early_random_commit":
            commitment_source, certified = "forced_early", False
        elif origin == "learned_commitment":
            commitment_source, certified = "learned_direct", False
        else:
            commitment_source, certified = "determined", True
        prediction["identity_commitment"] = {
            "source": commitment_source,
            "certified": certified,
            "mechanism": mechanism,
        }
        prediction["lowering"] = {
            "executed": True,
            "dsl_valid": True,
            "dsl": lowered.dsl,
        }
        prediction["predicted"] = {
            "specification": lowered.specification,
            "grounded_constraints": lowered.grounded_constraints,
        }
        route = backend(
            lowered.dsl,
            scene.asg,
            grounded_contract=lowered.grounded_constraints,
            robot_position=request.robot_position,
        )
        prediction["route"] = route
        found = route.get("status") == "SAT" and bool(route.get("waypoints"))
        prediction["execution"] = {
            "executed": True,
            "solver_success": found,
            "status": route.get("status"),
        }
        prediction["stage_verification"].update(
            dsl_valid=True, solver_success=found
        )
        if not found:
            self._failure(
                prediction,
                "route",
                route.get("status", "UNKNOWN"),
                route.get("status_reason"),
            )

    def _prepare_label_grounding(
        self,
        request,
        contract,
        prediction,
        scene,
        recorder,
        *,
        shared_label_grounding=None,
        grounding_policy: str = "determine_first",
    ):
        if contract is None:
            return
        assurance = audit_semantic_contract(request.instruction, contract)
        prediction["semantic_assurance"] = assurance.model_dump(mode="json")
        if not assurance.accepted:
            return
        if shared_label_grounding is not None:
            prediction["label_grounding"] = deepcopy(shared_label_grounding)
            prediction["label_grounding"]["shared_upstream"] = True
            prediction["label_grounding"].setdefault("grounding_policy", grounding_policy)
            return
        artifact, calls = self.compile_label_grounding(
            request,
            contract,
            scene,
            grounding_policy=grounding_policy,
        )
        prediction["label_grounding"] = deepcopy(artifact)
        if prediction["label_grounding"] is not None:
            prediction["label_grounding"]["shared_upstream"] = False
        # compile_label_grounding owns a separate recorder only in this fallback path.
        # Preserve the call records in the current prediction for accounting.
        for call in calls:
            recorder.calls.append(call)

    @staticmethod
    def _annotate_resolution_failure(prediction, resolver, exc):
        report = deepcopy(getattr(exc, "report", None) or {})
        if not report and resolver.history:
            report = deepcopy(resolver.history[-1])
        report.setdefault("status", "unresolved")
        report["replay_failure"] = {
            "code": getattr(exc, "code", None) or (
                "SOLVER_UNKNOWN"
                if isinstance(exc, QueryBudgetExhausted)
                else type(exc).__name__
            ),
            "message": str(exc),
            "infrastructure": bool(
                getattr(exc, "infrastructure", False)
                or isinstance(exc, QueryBudgetExhausted)
            ),
        }
        prediction["candidate_resolution"] = report
        history = deepcopy(resolver.history)
        if history and report.get("candidate_domains") is not None:
            history[-1] = deepcopy(report)
        elif report:
            history.append(deepcopy(report))
        prediction["candidate_resolution_history"] = history

    def _resolve_without_verification(
        self,
        request,
        contract,
        prediction,
        scene,
        recorder,
        backend,
        *,
        label_grounding_result=None,
        grounding_policy: str,
    ):
        """Run semantic parsing, label grounding and progressive commitment for No Verify/Repair."""
        resolver = ProgressiveResolver(
            backend, max_assignments=self.config.max_assignments
        )
        self._prepare_label_grounding(
            request,
            contract,
            prediction,
            scene,
            recorder,
            shared_label_grounding=label_grounding_result,
            grounding_policy=grounding_policy,
        )
        prediction["assurance"] = {"verification": None, "repair": None}
        prediction["method_policy"].update(
            {
                "grounding_policy": grounding_policy,
                "candidate_policy": "progressive_constraint_resolution",
                "verification_enabled": False,
                "repair_enabled": False,
                "commitment_rule": (
                    "local determination; joint hard feasibility; explicit preference; "
                    "otherwise unresolved"
                ),
            }
        )
        if contract is None:
            self._failure(
                prediction,
                "semantic",
                "SEMANTIC_CONTRACT_UNAVAILABLE",
                "semantic compilation did not produce a contract",
            )
            prediction["candidate_resolution_history"] = resolver.history
            return
        semantic_assurance = prediction.get("semantic_assurance")
        if (
            isinstance(semantic_assurance, dict)
            and semantic_assurance.get("accepted") is False
        ):
            self._failure(
                prediction,
                "semantic",
                "INVALID_SEMANTIC_CONTRACT",
                semantic_assurance.get("diagnostics", []),
            )
            prediction["candidate_resolution_history"] = resolver.history
            return
        label_grounding = prediction.get("label_grounding")
        validation = label_grounding.get("validation") if isinstance(label_grounding, dict) else None
        if not isinstance(validation, dict) or validation.get("success") is not True:
            self._failure(
                prediction,
                "grounding",
                "INVALID_LABEL_GROUNDING_GROUNDING",
                (
                    validation.get("diagnostics")
                    if isinstance(validation, dict)
                    else "validated label grounding artifact unavailable"
                ),
            )
            prediction["candidate_resolution_history"] = resolver.history
            return
        prediction["stage_verification"].update(
            json_valid=True,
            schema_valid=True,
        )
        try:
            bound, report = resolver(
                contract,
                label_grounding,
                scene.asg,
                request.robot_position,
            )
            prediction["candidate_resolution"] = deepcopy(report)
        except BindingResolutionError as exc:
            self._annotate_resolution_failure(prediction, resolver, exc)
            self._failure(
                prediction,
                "solver" if exc.infrastructure else "grounding",
                exc.code,
                str(exc),
            )
            return
        except QueryBudgetExhausted as exc:
            self._annotate_resolution_failure(prediction, resolver, exc)
            self._failure(
                prediction,
                "solver",
                "SOLVER_UNKNOWN",
                f"{type(exc).__name__}: {exc}",
            )
            return
        except Exception as exc:
            self._annotate_resolution_failure(prediction, resolver, exc)
            self._failure(
                prediction,
                "grounding",
                type(exc).__name__,
                f"{type(exc).__name__}: {exc}",
            )
            return
        try:
            self._execute_baseline(
                request,
                contract,
                bound,
                scene,
                backend,
                prediction,
                origin="progressive_constraint_resolution",
            )
        except QueryBudgetExhausted as exc:
            self._failure(
                prediction,
                "solver",
                "SOLVER_UNKNOWN",
                f"{type(exc).__name__}: {exc}",
            )
        except Exception as exc:
            stage = "solver" if prediction["lowering"].get("executed") else "lowering"
            self._failure(
                prediction,
                stage,
                type(exc).__name__,
                f"{type(exc).__name__}: {exc}",
            )
        prediction["candidate_resolution_history"] = deepcopy(resolver.history)

    def _forced_early(
        self,
        request,
        contract,
        prediction,
        scene,
        recorder,
        backend,
        *,
        label_grounding_result=None,
    ):
        """Forced-Early ablation with assurance before arbitrary commitment.

        Forced-Early shares semantic parsing, Determine-first label grounding, verification, and semantic/
        grounding repair with ReCoNav. It never uses hard/preference solver evidence
        to resolve a multi-ID identity. After semantic/grounding assurance is
        exhausted, any genuine remaining ambiguity is collapsed by a
        deterministic seeded pseudo-random choice.
        """
        self._prepare_label_grounding(
            request,
            contract,
            prediction,
            scene,
            recorder,
            shared_label_grounding=label_grounding_result,
        )
        prediction["method_policy"].update(
            {
                "grounding_policy": "determine_first",
                "candidate_policy": "post_assurance_seeded_random_commit",
                "verification_enabled": True,
                "repair_enabled": True,
                "forced_early_unique_binding": True,
                "global_constraint_candidate_filtering": False,
                "repair_direct_instance_selection": False,
                "learned_candidate_selector": False,
            }
        )

        resolver = ForcedEarlyResolver()
        outcome = run_verification(
            instruction=request.instruction,
            robot_position=request.robot_position,
            artifact=prediction,
            asg=scene.asg,
            force_recompute=True,
            binding_resolver=resolver,
            execution_backend=backend,
        )
        before = self._outcome_summary(outcome)
        rounds: list[dict[str, Any]] = []
        semantic_repair_resolver = ForcedEarlyResolver()
        model_calls = 0

        for round_number in range(1, self.config.repair_rounds + 1):
            if outcome.verification_accepted or outcome.needs_backend_fix:
                break

            # Ambiguity may be caused by incorrect semantics/grounding, so
            # Forced-Early receives the same bounded semantic/grounding repair as ReCoNav.
            # Pure route infeasibility is never permission to rewrite semantics.
            blocked = [
                diagnostic
                for diagnostic in outcome.diagnostics_after_repair
                if diagnostic.severity == "BLOCK"
            ]
            if not any(
                diagnostic.category in {
                    "ARTIFACT",
                    "COMMITMENT",
                    "GROUNDING",
                }
                for diagnostic in blocked
            ):
                break

            semantic_repair = run_semantic_repair(
                instruction=request.instruction,
                robot_position=request.robot_position,
                starting_outcome=outcome,
                asg=scene.asg,
                call_model=self._repair_call(recorder),
                max_rounds=1,
                include_warnings=False,
                binding_resolver=semantic_repair_resolver,
                execution_backend=backend,
                preflight=self._repair_preflight(recorder),
            )
            model_calls += semantic_repair.model_calls
            rounds.extend(
                {
                    **dict(item),
                    "outer_round": round_number,
                }
                for item in semantic_repair.rounds
            )
            next_outcome = semantic_repair.final_outcome
            if next_outcome is outcome and not semantic_repair.rounds:
                break
            outcome = next_outcome

        forced_record = None
        forced_history: list[dict[str, Any]] = []

        # Only after verification/repair has exhausted semantic/grounding recovery does the
        # ablation discard a genuine residual candidate set.
        if self._identity_ambiguity(outcome):
            artifact = deepcopy(outcome.prediction_artifact)
            report = artifact.get("candidate_resolution")
            choices, forced_record = seeded_random_identity_choice(
                report if isinstance(report, Mapping) else {},
                seed=self.config.seed,
                scene_id=request.scene_id,
                task_id=request.task_id,
            )
            if choices is not None:
                pinned = PinnedIdentityResolver(
                    choices,
                    mechanism="forced_early_random_commit",
                )
                outcome = run_verification(
                    instruction=request.instruction,
                    robot_position=request.robot_position,
                    artifact=artifact,
                    asg=scene.asg,
                    force_recompute=True,
                    binding_resolver=pinned,
                    execution_backend=backend,
                )
                forced_history = deepcopy(pinned.history)

        prediction.update(outcome.prediction_artifact)
        prediction["assurance"] = {
            "before_repair": before,
            "final": self._outcome_summary(outcome),
            "repair": {
                "rounds": rounds,
                "stopped_reason": (
                    "accepted"
                    if outcome.verification_accepted
                    else "backend/infrastructure"
                    if outcome.needs_backend_fix
                    else "budget exhausted or unresolved"
                ),
                "model_calls": model_calls,
            },
            "forced_commit_after_repair": deepcopy(forced_record),
        }
        prediction["candidate_resolution_history"] = [
            *deepcopy(resolver.history),
            *deepcopy(semantic_repair_resolver.history),
            *forced_history,
        ]
        if forced_record is not None:
            prediction["forced_early_choice"] = deepcopy(forced_record)
        prediction["method_policy"].update(
            {
                "candidate_policy": "post_assurance_seeded_random_commit",
                "commitment_rule": (
                    "share semantic/grounding assurance with ReCoNav; after repair, "
                    "collapse any genuine residual multi-ID frontier by a "
                    "deterministic seeded pseudo-random choice before global "
                    "constraint evidence may resolve identity"
                ),
                "forced_early_unique_binding": True,
                "global_constraint_candidate_filtering": False,
                "repair_direct_instance_selection": False,
                "learned_candidate_selector": False,
            }
        )
        self._refresh_identity_commitment(prediction)

    @staticmethod
    def _refresh_identity_commitment(prediction) -> None:
        report = prediction.get("candidate_resolution")
        mechanism = report.get("mechanism") if isinstance(report, Mapping) else None
        fallback = prediction.get("execution_fallback")
        if (
            isinstance(fallback, Mapping)
            and fallback.get("status") == "resolved"
            and isinstance(fallback.get("validated_choices"), Mapping)
        ):
            source, certified = "fallback", False
        elif mechanism == "forced_early_random_commit":
            source, certified = "forced_early", False
        elif mechanism in {
            "local_unique",
            "hard_constraint_resolved",
            "preference_selected",
        }:
            source, certified = "determined", True
        elif prediction.get("learned_decision") is not None:
            source, certified = "learned_direct", False
        else:
            source, certified = "unresolved", False
        prediction["identity_commitment"] = {
            "source": source,
            "certified": certified,
            "mechanism": mechanism,
        }

    @staticmethod
    def _final_candidate_choice_needed(
        prediction: Mapping[str, Any],
    ) -> bool:
        """Allow one execution-only tie-break for unresolved multi-ID resolution."""
        report = prediction.get("candidate_resolution")
        if not isinstance(report, Mapping):
            return False
        raw_domains = report.get("ambiguous_domains")
        if not isinstance(raw_domains, Mapping):
            raw_domains = report.get("candidate_domains")
        has_multi = (
            isinstance(raw_domains, Mapping)
            and any(
                isinstance(values, list) and len(values) > 1
                for values in raw_domains.values()
            )
        )
        if not has_multi:
            return False

        status = str(report.get("status", ""))
        if status == "ambiguous":
            return True
        if status != "unknown":
            return False

        replay_failure = report.get("replay_failure")
        replay_code = (
            replay_failure.get("code")
            if isinstance(replay_failure, Mapping)
            else None
        )
        failure = prediction.get("failure")
        failure_reason = (
            failure.get("reason")
            if isinstance(failure, Mapping)
            else None
        )
        return (
            replay_code == "SOLVER_UNKNOWN"
            or failure_reason == "SOLVER_UNKNOWN"
        )

    @staticmethod
    def _identity_ambiguity(outcome: VerificationOutcome) -> bool:
        """True only when unresolved multi-ID identity is the sole verification BLOCK."""
        blocked = [
            diagnostic
            for diagnostic in outcome.diagnostics_after_repair
            if diagnostic.severity == "BLOCK"
        ]
        report = outcome.prediction_artifact.get("candidate_resolution")
        if not isinstance(report, Mapping):
            return False
        raw_domains = report.get("ambiguous_domains")
        if not isinstance(raw_domains, Mapping):
            raw_domains = report.get("candidate_domains")
        has_multi = (
            isinstance(raw_domains, Mapping)
            and any(
                isinstance(values, list) and len(values) > 1
                for values in raw_domains.values()
            )
        )
        return has_multi and bool(blocked) and all(
            diagnostic.code == "IDENTITY_REQUIRED_AMBIGUOUS"
            for diagnostic in blocked
        )

    def _apply_final_identity_fallback(
        self,
        *,
        request,
        scene,
        prediction,
        recorder,
        backend,
    ) -> bool:
        """Execution-only seeded tie-break after progressive evidence is exhausted.

        No extra LLM candidate-selection call is made. ReCoNav/No-Verify-Repair/Before-Repair
        first use their ordinary progressive evidence; if identity remains
        unresolved, the same deterministic pseudo-random mechanism as Forced-Early
        chooses only among assignments not excluded by that evidence.
        """
        del recorder
        semantic_contract = prediction.get("semantic_contract")
        label_grounding = prediction.get("label_grounding")
        if not isinstance(semantic_contract, dict) or not isinstance(label_grounding, dict):
            return False
        if not self._final_candidate_choice_needed(prediction):
            return False

        original_report = prediction.get("candidate_resolution")
        choices, record = seeded_random_identity_choice(
            original_report if isinstance(original_report, Mapping) else {},
            seed=self.config.seed,
            scene_id=request.scene_id,
            task_id=request.task_id,
            prefer_proved_feasible=True,
        )
        prediction["execution_fallback"] = deepcopy(record)
        prediction["method_policy"]["final_execution_fallback"] = True
        prediction["method_policy"]["final_execution_fallback_policy"] = (
            "seeded_random_from_progressive_admissible_set"
        )
        if choices is None:
            return False

        try:
            contract = SceneIndependentSemanticContract.model_validate(
                semantic_contract
            )
            pinned = PinnedIdentityResolver(
                choices,
                mechanism="fallback_seeded_random",
            )
            bound, report = pinned(
                contract,
                label_grounding,
                scene.asg,
                request.robot_position,
            )
            prediction.setdefault(
                "candidate_resolution_history", []
            ).extend(deepcopy(pinned.history))
            prediction["certified_candidate_resolution"] = deepcopy(
                original_report
            )
            prediction["fallback_candidate_resolution"] = deepcopy(report)
            prediction["candidate_resolution"] = deepcopy(report)
            prediction["certification_failure"] = deepcopy(
                prediction.get("failure")
            )
            prediction["failure"] = None
            self._execute_baseline(
                request,
                contract,
                bound,
                scene,
                backend,
                prediction,
                origin="fallback_random",
            )
            return prediction.get("failure") is None
        except Exception as exc:
            prediction["execution_fallback"]["execution_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            return False

    def _repair_call(self, recorder):
        def call(system, user, fmt):
            with recorder.at("semantic_repair"):
                raw = recorder.generate(
                    system,
                    user,
                    response_format=fmt,
                    **self.config.generation_kwargs(),
                )
            return raw, {"record_index": len(recorder.calls) - 1}

        return call

    def _repair_preflight(self, recorder):
        return lambda system, user, fmt: recorder.preflight(
            system,
            user,
            fmt,
            max_tokens=self.config.max_tokens,
            enable_thinking=self.config.enable_thinking,
        )

    @staticmethod
    def _outcome_summary(outcome: VerificationOutcome) -> dict[str, Any]:
        return {
            key: value
            for key, value in outcome.model_dump(mode="json").items()
            if key != "prediction_artifact"
        }

    @staticmethod
    def _exception_failure_record(exc):
        if isinstance(exc, ContextBudgetError):
            stage, reason = "model_input", "CONTEXT_BUDGET_EXCEEDED"
        elif isinstance(exc, InputCharacterGuardError):
            stage, reason = "model_input", "INPUT_CHARACTER_GUARD_EXCEEDED"
        elif isinstance(exc, InputBudgetError):
            stage, reason = "model_input", "INPUT_BUDGET_EXCEEDED"
        elif isinstance(exc, QueryBudgetExhausted):
            stage, reason = "solver", "SOLVER_UNKNOWN"
        else:
            stage, reason = "model_response", "UPSTREAM_INFRASTRUCTURE_FAILURE"
        return {
            "stage": stage,
            "reason": reason,
            "detail": f"{type(exc).__name__}: {exc}",
        }

    def _classify_unhandled_failure(self, prediction, exc):
        if prediction["lowering"].get("executed") and not isinstance(
            exc,
            (ContextBudgetError, InputCharacterGuardError, InputBudgetError),
        ):
            failure = {
                "stage": "solver",
                "reason": (
                    "SOLVER_UNKNOWN"
                    if isinstance(exc, QueryBudgetExhausted)
                    else "BACKEND_ERROR"
                ),
                "detail": f"{type(exc).__name__}: {exc}",
            }
        elif prediction.get("learned_decision") and not isinstance(
            exc,
            (ContextBudgetError, InputCharacterGuardError, InputBudgetError),
        ):
            failure = {
                "stage": "grounding",
                "reason": "INVALID_LEARNED_CONTRACT",
                "detail": f"{type(exc).__name__}: {exc}",
            }
        else:
            failure = self._exception_failure_record(exc)
        self._failure(
            prediction,
            failure["stage"],
            failure["reason"],
            failure["detail"],
        )

    def _recorder(self):
        return RecordedClient(
            self.client,
            max_input_characters=self.config.max_input_characters,
            tokenizer_name_or_path=self.config.tokenizer_name_or_path,
            served_max_model_len=self.config.served_max_model_len,
            context_safety_margin=self.config.context_safety_margin,
        )
