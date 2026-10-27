"""Paired experiments: prediction receives a whitelist; evaluation receives gold later."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
import copy
import hashlib
import json
import os
import multiprocessing
import shutil
import subprocess
import time
from datetime import datetime, timezone
from urllib.request import Request, urlopen

from reconav.benchmark.metrics import evaluate_prediction, grounded_gold, replayable_route, summarize_predictions
from reconav.benchmark.route_evaluator import evaluate_grounded_route
from reconav.llm.shared import SharedLLMClient
from reconav.solver.query_evidence import FEASIBLE
from reconav.method.runtime.backend import BoundedBackend
from reconav.method.runtime.accounting import infrastructure_errors
from reconav.method.runtime.models import (
    ABLATION_METHODS,
    MAIN_METHODS,
    DirectAttemptResult,
    DirectDecision,
    NavigationRequest,
    RuntimeConfig,
    METHODS,
    RUNTIME_VERSION,
    method_name,
    method_backbone,
    normalize_method,
    method_provider,
    method_variant,
    methods_for_provider,
)
from reconav.method.runtime.pipeline import NavigationPipeline
from reconav.method.runtime.result_layout import prediction_path, prediction_paths, result_method_id
from reconav.method.runtime.prompts import (
    direct_feedback_prompt,
    direct_prompt,
    final_fallback_binding_prompt,
    forced_early_binding_prompt,
    direct_id_sequence_prompt,
)
from reconav.method.runtime.scene import DirectorySceneProvider, digest
from reconav.method.semantic.compiler import SemanticCompilationResult, load_system_prompt
from reconav.method.semantic.contracts import SceneIndependentSemanticContract
from reconav.llm.control import ProviderRunStopped, check_provider_stop, provider_run_guard, stop_record


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def repository_identity():
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    hasher = hashlib.sha256()
    # Version the method and shared semantics actually consumed. Unrelated asset
    # edits do not make a physical cache private to this experiment.
    for folder in ("method", "solver", "benchmark", "scene"):
        for p in sorted((root / folder).rglob("*.py")):
            hasher.update(p.relative_to(root).as_posix().encode()); hasher.update(p.read_bytes())
    for relative in ("solver/backend.py", "llm/client.py", "llm/control.py", "scene/navigation.py", "scene/mesh_physical.py", "scene/mesh_io.py", "scene/io.py", "scene/vocabulary.py",
                     "scene/semantic_policy.json",
                     "commands/navigation.py"):
        p = root / relative
        hasher.update(relative.encode()); hasher.update(p.read_bytes())
    for folder in ("prompts/scene_independent_semantics", "prompts/semantic_grounding", "prompts/semantic_repair"):
        for p in sorted((root / folder).rglob("*.txt")):
            hasher.update(p.relative_to(root).as_posix().encode()); hasher.update(p.read_bytes())
    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "dirty": bool(git("status", "--porcelain")), "runtime_source_sha256": hasher.hexdigest()}


def qwen_preflight(base_url: str, model: str | None, timeout: float = 15.0):
    headers = {}
    if os.environ.get("QWEN_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["QWEN_API_KEY"]
    request = Request(base_url.rstrip("/") + "/models", headers=headers)
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    rows = payload.get("data", [])
    if model is None:
        if len(rows) != 1:
            raise ValueError("Qwen service exposes zero or multiple models; specify --qwen-model explicitly")
        model = str(rows[0]["id"])
    chosen = next((row for row in rows if row.get("id") == model), None)
    if chosen is None:
        raise ValueError(f"requested Qwen model {model!r} is not served")
    return {"provider": "qwen", "endpoint": base_url, "served_model": model,
            "runtime_model": {key: chosen[key] for key in ("id", "root", "parent", "max_model_len") if key in chosen}}


def load_tasks(root: Path, scenes=(), task_ids=(), limit: int | None = None):
    root = root.resolve()
    scene_dirs = [root / scene for scene in scenes] if scenes else sorted(p for p in root.iterdir() if p.is_dir())
    wanted = set(task_ids)
    result = []
    found = set()
    for scene_dir in scene_dirs:
        if scene_dir.name in {".", ".."} or scene_dir.parent != root:
            raise ValueError("scene IDs must be direct subdirectories")
        files = sorted(scene_dir.glob("*.json"))
        if scenes and not files:
            raise FileNotFoundError(f"no tasks in {scene_dir}")
        for path in files:
            task = read_json(path)
            if wanted and task.get("task_id") not in wanted:
                continue
            request = NavigationRequest.from_task(task)
            if request.scene_id != scene_dir.name or path.stem != request.task_id:
                raise ValueError(f"task filename/scene identity mismatch: {path}")
            result.append((path, task, request)); found.add(request.task_id)
    if wanted - found:
        raise ValueError(f"missing requested tasks: {sorted(wanted-found)}")
    result = result if limit is None else result[:limit]
    if not result:
        raise ValueError("empty task selection")
    return result


def select_tasks(tasks_root, scenes=(), task_ids=(), limit=None, task_order="mechanism-first"):
    tasks = load_tasks(Path(tasks_root), scenes, task_ids)
    if task_order not in {"mechanism-first", "filename"}:
        raise ValueError("unsupported task order")
    if task_order == "mechanism-first":
        # Scheduler metadata never enters the predictor's NavigationRequest.
        # Prioritize the mechanism comparison across all selected scenes.
        tasks.sort(key=lambda item: (
            (item[1].get("uniqueness_proof") or {}).get("identity_class")
            not in {"hard_constraint", "soft_preference"},
            -len((item[1].get("symbolic_ir") or {}).get("constraints", [])),
            -len((item[1].get("uniqueness_proof") or {}).get("qualified_candidate_ids", [])),
            item[2].scene_id, item[2].task_id,
        ))
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def _compile_record(result, calls, fingerprint):
    return {"fingerprint": fingerprint, "raw_response": result.raw_response, "candidate": result.candidate,
            "parsed_output": result.parsed_output.model_dump(mode="json") if result.parsed_output else None,
            "parse_error": result.parse_error, "infrastructure_error": result.infrastructure_error,
            "timing": result.timing, "model_calls": calls}


def _restore_semantic(record):
    parsed = record.get("parsed_output")
    return SemanticCompilationResult(record.get("raw_response", ""), record.get("candidate"),
        SceneIndependentSemanticContract.model_validate(parsed) if parsed is not None else None,
        record.get("parse_error"), record.get("timing"), record.get("infrastructure_error"))


def _direct_record(result, calls, fingerprint):
    return {
        "fingerprint": fingerprint,
        "raw_response": result.raw_response,
        "parsed_output": (
            result.parsed_output.model_dump(mode="json")
            if result.parsed_output is not None
            else None
        ),
        "parse_error": result.parse_error,
        "infrastructure_error": result.infrastructure_error,
        "timing": result.timing,
        "model_calls": calls,
    }


def _restore_direct(record):
    parsed = record.get("parsed_output")
    return DirectAttemptResult(
        record.get("raw_response", ""),
        DirectDecision.model_validate(parsed) if parsed is not None else None,
        record.get("parse_error"),
        record.get("infrastructure_error"),
        record.get("timing"),
    )


def _runtime_prediction_snapshot(record):
    """Strip post-prediction evaluation metadata before method reuse."""
    snapshot = copy.deepcopy(record)
    for key in (
        "metrics",
        "evaluation_status",
        "evaluation_error",
        "input_fingerprint",
        "run_id",
        "run_directory",
        "result_method_id",
        "task_type",
        "difficulty",
    ):
        snapshot.pop(key, None)
    return snapshot


def _runtime_prediction_digest(record):
    payload = json.dumps(
        _runtime_prediction_snapshot(record),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def evaluate_reference(task, scene, seconds):
    """Post-prediction gold executability evidence.

    Metrics score the executed route directly against gold constraints, so
    preference optimization is not part of evaluation. The reference pass only
    checks that the same-source gold task has a hard-feasible witness.
    """
    record = {"status": "not_checked"}
    dsl = task.get("grounded_dsl")
    if not isinstance(dsl, str) or not dsl.strip():
        return {
            **record,
            "status": "reference_missing",
            "reason": "no grounded_dsl in task",
        }
    try:
        fixed = grounded_gold(task)
        backend = BoundedBackend(
            per_query_seconds=seconds,
            total_seconds=seconds,
        )
        route = backend(
            dsl,
            scene.asg,
            request=FEASIBLE,
        )
        record.update(
            solver_status=route.get("status"),
            solver_reason=route.get("status_reason"),
            query_outcome=route.get("query_outcome"),
            proof_domain=route.get("proof_domain"),
        )
        if route.get("status") != "SAT" or not route.get("waypoints"):
            return {
                **record,
                "status": (
                    "reference_unknown"
                    if route.get("status") == "UNKNOWN"
                    else "reference_invalid"
                ),
                "queries": backend.queries,
            }

        replay, errors = replayable_route(
            route,
            fixed,
            scene.asg,
            task["robot_position"],
        )
        verification = evaluate_grounded_route(
            scene.asg,
            fixed,
            replay,
            task["robot_position"],
        )
        hard_errors = [
            item
            for item in verification.get("constraints", [])
            if not item.get("satisfied")
            and item.get("component")
            not in {"prefer", "prefer_order", "preference_optimality"}
        ]
        if errors or hard_errors:
            return {
                **record,
                "status": "reference_invalid",
                "route_errors": errors,
                "hard_errors": hard_errors,
                "queries": backend.queries,
            }

        return {
            **record,
            "status": "valid",
            "verification": verification,
            "queries": backend.queries,
        }
    except Exception as exc:
        return {
            **record,
            "status": "reference_error",
            "reason": f"{type(exc).__name__}: {exc}",
        }


def summarize_run(run_dir: Path, methods):
    llm_provider = method_provider(methods[0])
    m = lambda variant: method_name(method_backbone(methods[0]), variant)
    manifest = read_json(run_dir / "run_manifest.json")
    outputs = {m: [read_json(p) for p in prediction_paths(run_dir, m, manifest)] for m in methods}
    metric_summaries = {m: summarize_predictions(rows) for m, rows in outputs.items()}
    expected = set(manifest.get("tasks", {}))
    health = {
        method: {
            "expected_tasks": len(expected),
            "missing_tasks": sorted(expected - {f"{r['scene_id']}/{r['task_id']}" for r in rows}),
            "unexpected_tasks": sorted({f"{r['scene_id']}/{r['task_id']}" for r in rows} - expected),
            "infrastructure_errors": sum(bool(infrastructure_errors(r)) for r in rows),
            "evaluation_errors": sum(r.get("evaluation_status") != "evaluated" for r in rows),
        } for method, rows in outputs.items()
    }
    valid_run = all(not row["missing_tasks"] and not row["unexpected_tasks"]
                    and not row["infrastructure_errors"] and not row["evaluation_errors"] for row in health.values())
    if not valid_run:
        for scores in metric_summaries.values():
            scores["valid"] = False
            for value in scores.values():
                if isinstance(value, dict) and "rate" in value:
                    value["rate"] = None
    model_info = manifest.get("settings", {}).get("model") or {}
    backbone = model_info.get("served_model") or model_info.get("model") or llm_provider
    def display_name(method):
        prefix = "Gemma" if "gemma" in str(backbone).lower() else method.split("-", 1)[0]
        variant = method_variant(method)
        return prefix + "-" + variant
    main_methods = [method for method in methods if method in MAIN_METHODS]
    ablation_methods = [method for method in methods if method in ABLATION_METHODS]
    summary = {"active_methods": list(methods), "valid_for_reporting": valid_run, "health": health,
        "backbone": backbone, "method_display_names": {m: display_name(m) for m in methods},
        "result_method_ids": {m: result_method_id(m, manifest.get("settings", {}).get("asg_data_source", "hm3dsem")) for m in methods},
        "paper_main_methods": main_methods,
        "ablation_methods": ablation_methods,
        "run_id": run_dir.name,
        "task_count_per_method": {m: len(rows) for m, rows in outputs.items()},
        "original_six_metrics": metric_summaries,
        "main_table_metrics": {
            method: {
                key: metric_summaries[method][key]
                for key in ("full_is", "hard_ns", "preference_satisfaction")
            }
            for method in main_methods
        },
        "ablation_metrics": {
            method: {
                key: metric_summaries[method][key]
                for key in ("full_is", "hard_ns", "preference_satisfaction")
            }
            for method in ablation_methods
        },
        "failures": {m: dict(Counter((r.get("failure") or {}).get("reason", "none") for r in rows)) for m, rows in outputs.items()},
        "evaluation_statuses": {m: dict(Counter(r.get("evaluation_status", "pending") for r in rows)) for m, rows in outputs.items()},
        "timing": {m: {"pipeline_seconds": sum(r.get("timing", {}).get("pipeline_seconds", 0) for r in rows),
                       "own_model_calls": sum(r.get("call_accounting", {}).get("own_model_calls", 0) for r in rows),
                       "logical_calls_including_reused_upstream": sum(
                           r.get("call_accounting", {}).get(
                               "total_logical_calls_including_reused_upstream",
                               r.get("call_accounting", {}).get(
                                   "total_logical_calls_including_reused_semantic", 0
                               ),
                           )
                           for r in rows
                       )} for m, rows in outputs.items()},
        "denominators": (
            "Hard NS and Full IS are scored on every evaluated task. Preference "
            "Satisfaction is scored only on tasks with explicit preferences and is "
            "the fraction of those clauses satisfied by the executed route. No "
            "preference oracle optimization is used by the metric."
        )}
    by_task = {}
    for method, rows in outputs.items():
        for row in rows:
            by_task.setdefault((row["scene_id"], row["task_id"]), {})[method] = {
                "full_is": row.get("metrics", {}).get("full_is"), "failure": row.get("failure"),
                "selection": row.get("candidate_resolution", {}).get("mechanism")}
    summary["paired_tasks"] = [{"scene_id": scene, "task_id": task, "methods": values} for (scene, task), values in sorted(by_task.items())]

    def first_candidate_domains(row):
        for report in row.get("candidate_resolution_history", []):
            domains = report.get("candidate_domains")
            if isinstance(domains, dict):
                return {
                    str(path): tuple(int(value) for value in values)
                    for path, values in domains.items()
                    if isinstance(values, list)
                }
        return None

    direct_method = m("Direct")
    feedback_method = m("Direct-Feedback")
    if direct_method in outputs and feedback_method in outputs:
        direct_rows = {
            (row["scene_id"], row["task_id"]): row
            for row in outputs[direct_method]
        }
        feedback_rows = {
            (row["scene_id"], row["task_id"]): row
            for row in outputs[feedback_method]
        }
        comparable = 0
        exact = 0
        mismatches = []
        for key in sorted(set(direct_rows) & set(feedback_rows)):
            direct_sha = (
                direct_rows[key]
                .get("direct_pairing", {})
                .get("complete_first_prediction_sha256")
            )
            feedback = feedback_rows[key].get("direct_feedback", {})
            feedback_sha = feedback.get("shared_first_prediction_sha256")
            attempts = feedback.get("attempts", [])
            first = attempts[0] if attempts else {}
            if not direct_sha or not feedback_sha:
                continue
            comparable += 1
            matched = (
                direct_sha == feedback_sha
                and first.get("is_shared_direct_first_attempt") is True
                and first.get("reused_complete_direct_result") is True
                and first.get("success")
                is (direct_rows[key].get("failure") is None)
            )
            if matched:
                exact += 1
            else:
                mismatches.append({
                    "scene_id": key[0],
                    "task_id": key[1],
                    "direct_sha": direct_sha,
                    "feedback_sha": feedback_sha,
                    "first_attempt": first,
                })
        summary["direct_feedback_pairing"] = {
            "comparable_tasks": comparable,
            "exact_complete_first_result_reuse": exact,
            "all_exact": comparable > 0 and exact == comparable,
            "mismatches": mismatches,
        }

    reconav_method = m("ReCoNav")
    forced_method = m("Forced-Early")
    if reconav_method in outputs and forced_method in outputs:
        reconav_rows = {
            (row["scene_id"], row["task_id"]): row
            for row in outputs[reconav_method]
        }
        forced_rows = {
            (row["scene_id"], row["task_id"]): row
            for row in outputs[forced_method]
        }
        comparable = 0
        equal = 0
        mismatches = []
        for key in sorted(set(reconav_rows) & set(forced_rows)):
            reconav_domains = first_candidate_domains(reconav_rows[key])
            forced_domains = first_candidate_domains(forced_rows[key])
            if reconav_domains is None or forced_domains is None:
                continue
            comparable += 1
            if reconav_domains == forced_domains:
                equal += 1
            else:
                mismatches.append({
                    "scene_id": key[0],
                    "task_id": key[1],
                    "reconav": reconav_domains,
                    "forced_early": forced_domains,
                })
        summary["forced_early_pairing"] = {
            "comparable_tasks": comparable,
            "identical_initial_candidate_domains": equal,
            "all_identical": comparable > 0 and equal == comparable,
            "mismatches": mismatches,
        }

    progressive_methods = tuple(m(v) for v in ("No-Verify-Repair", "Before-Repair", "ReCoNav") if m(v) in outputs)
    if len(progressive_methods) >= 2:
        indexed = {
            method: {
                (row["scene_id"], row["task_id"]): row
                for row in outputs[method]
            }
            for method in progressive_methods
        }
        keys = set.intersection(*(set(rows) for rows in indexed.values()))
        comparable = 0
        valid = 0
        mismatches = []
        expected_stage = {
            m("No-Verify-Repair"): "No-Verify-Repair",
            m("Before-Repair"): "Before-Repair",
            m("ReCoNav"): "ReCoNav",
        }
        for key in sorted(keys):
            metadata = {
                method: indexed[method][key].get("progressive_frontier", {})
                for method in progressive_methods
            }
            comparable += 1
            matched = all(
                item.get("single_run") is True
                and item.get("stage") == expected_stage[method]
                and item.get("sequence") == ["No-Verify-Repair", "Before-Repair", "ReCoNav"]
                for method, item in metadata.items()
            )
            if matched:
                valid += 1
            else:
                mismatches.append({
                    "scene_id": key[0],
                    "task_id": key[1],
                    "frontiers": metadata,
                })
        summary["progressive_ablation_pairing"] = {
            "methods": list(progressive_methods),
            "construction": "single_progressive_frontier",
            "comparable_tasks": comparable,
            "valid_single_frontier_tasks": valid,
            "all_identical": comparable > 0 and valid == comparable,
            "mismatches": mismatches,
        }

    summary["logical_cost"] = {
        method: {key: sum(float(r.get("logical_cost", {}).get(key, 0)) for r in rows)
                 for key in ("model_calls", "input_tokens", "output_tokens", "billable_output_tokens", "total_tokens", "cached_input_tokens", "cache_write_input_tokens", "reasoning_tokens", "llm_seconds", "solver_queries", "solver_seconds")}
        for method, rows in outputs.items()
    }
    # Count physical requests by identity, not by prompt hash: repeated identical
    # requests cost money too. Shared copies retain their original call_id.
    billed = {}
    def collect(value):
        if isinstance(value, dict):
            if value.get("call_id") and "provider_called" in value:
                billed[value["call_id"]] = value
                return
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
    collect(outputs)
    for path in sorted((run_dir / "scenes").rglob("*.json")):
        collect(read_json(path))
    for path in sorted((run_dir / "infrastructure_attempts").rglob("*.json")):
        collect(read_json(path))
    physical_calls = [call for call in billed.values() if call.get("provider_called")]
    summary["actual_model_cost"] = {
        "requests": len(physical_calls),
        "transport_attempts": sum(len(call.get("transport_attempts", [])) for call in physical_calls),
        "requests_without_usage": sum(not call.get("usage", {}).get("total_tokens") for call in physical_calls),
        **{key: sum(float(call.get("usage", {}).get(key, 0)) for call in physical_calls)
           for key in ("input_tokens", "output_tokens", "billable_output_tokens", "total_tokens", "cached_input_tokens", "cache_write_input_tokens", "reasoning_tokens")},
        "includes_archived_infrastructure_attempts": True,
    }
    write_json(run_dir / "summary.json", summary)
    def render(method, key):
        score = summary["original_six_metrics"][method][key]
        if score["rate"] is None:
            return "n/a"
        return f"{score['rate']:.3f} ({score['applicable']})"

    markdown = [
        f"# {run_dir.name}", "",
        f"Backbone: {backbone}. Valid for reporting: {valid_run}.",
        "",
        "## Main table",
        "",
        "| Method | Tasks | Full IS | Hard NS | Pref. Sat. |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in main_methods:
        markdown.append(
            f"| {display_name(method)} | {len(outputs[method])} | "
            f"{render(method, 'full_is')} | "
            f"{render(method, 'hard_ns')} | "
            f"{render(method, 'preference_satisfaction')} |"
        )
    if ablation_methods:
        markdown.extend([
            "",
            "## Ablation",
            "",
            "| Method | Tasks | Full IS | Hard NS | Pref. Sat. |",
            "|---|---:|---:|---:|---:|",
        ])
        for method in ablation_methods:
            markdown.append(
                f"| {display_name(method)} | {len(outputs[method])} | "
                f"{render(method, 'full_is')} | "
                f"{render(method, 'hard_ns')} | "
                f"{render(method, 'preference_satisfaction')} |"
            )
    (run_dir / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return summary



_TASK_PROCESS_PROVIDER = None
_TASK_PROCESS_PIPELINE = None


def _initialize_task_process(
    asg_root: str,
    asg_data_source: str,
    config: RuntimeConfig,
    llm_provider: str,
):
    """Initialize one independent task worker process.

    Each process owns its scene cache, tokenizer cache, pipeline and solver state.
    Every learned call in the worker uses the run's single LLM provider.
    """
    global _TASK_PROCESS_PROVIDER, _TASK_PROCESS_PIPELINE
    _TASK_PROCESS_PROVIDER = DirectorySceneProvider(Path(asg_root), asg_data_source)
    _TASK_PROCESS_PIPELINE = NavigationPipeline(
        SharedLLMClient(llm_provider), config
    )


def _run_task_process(job):
    if _TASK_PROCESS_PROVIDER is None or _TASK_PROCESS_PIPELINE is None:
        raise RuntimeError("task process was not initialized")
    task_job = dict(job)
    qwen_base_url = task_job.pop("qwen_base_url", None)
    if qwen_base_url is not None:
        os.environ["QWEN_BASE_URL"] = str(qwen_base_url)
    return _run_one_task(
        provider=_TASK_PROCESS_PROVIDER,
        pipeline=_TASK_PROCESS_PIPELINE,
        **task_job,
    )


def _run_one_task(**kwargs):
    """Retry infrastructure failures as a whole paired task, preserving evidence.

    Model parse/semantic/solver failures are outcomes, never selected retries.
    Evaluation-only failures reuse predictions and are reevaluated below.
    """
    output, request = Path(kwargs["output"]), kwargs["request"]
    scene_root = output / "scenes" / request.scene_id
    paths = lambda: [prediction_path(output, m, request.scene_id, request.task_id,
                                    results_root=kwargs.get("results_root"),
                                    asg_data_source=kwargs.get("asg_data_source", "hm3dsem"))
                     for m in kwargs["methods"]]
    def failed():
        return any(infrastructure_errors(read_json(p)) for p in paths() if p.exists())
    def archive():
        from uuid import uuid4
        destination = output / "infrastructure_attempts" / f"{request.scene_id}_{request.task_id}_{uuid4().hex}"
        for source in sorted(scene_root.rglob(f"{request.task_id}.json")):
            target = destination / source.relative_to(scene_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
        if kwargs.get("results_root") is not None:
            for method, source in zip(kwargs["methods"], paths()):
                if source.exists():
                    target = destination / "predictions" / method / f"{request.task_id}.json"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source), str(target))
    # Validate cached identity even when an old infrastructure failure is retried.
    fingerprint = digest([asdict(request), kwargs["expected_scene_fingerprint"], kwargs["config_hash"]])
    for path in paths():
        if path.exists() and read_json(path).get("input_fingerprint") != fingerprint:
            raise ValueError(f"cached prediction input mismatch: {path}")
    for attempt in range(3):
        check_provider_stop(method_provider(kwargs["methods"][0]))
        if failed():
            archive()
        result = _run_one_task_attempt(**kwargs)
        check_provider_stop(method_provider(kwargs["methods"][0]))
        if not failed():
            return result
        print(f"INFRASTRUCTURE {request.task_id} task_attempt={attempt + 1}/3", flush=True)
    return result


def _run_one_task_attempt(*, provider, pipeline, path, task, request, expected_scene_fingerprint,
                  output, methods, config_hash, task_data_source, asg_data_source,
                  prediction_only, reference_seconds, method_workers, results_root=None):
    """Run one task atomically with no cross-task mutable method state."""
    path = Path(path)
    output = Path(output)
    task_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    scene = provider.get(request.scene_id)
    if scene.fingerprint != expected_scene_fingerprint:
        raise ValueError(f"ASG changed while task was queued: {request.scene_id}")
    fingerprint = digest([asdict(request), scene.fingerprint, config_hash])
    llm_provider = method_provider(methods[0])
    if any(method_backbone(method) != method_backbone(methods[0]) for method in methods):
        raise ValueError("one task run may not mix model backbones")
    m = lambda variant: method_name(method_backbone(methods[0]), variant)
    target_for = lambda method: prediction_path(output, method, request.scene_id, request.task_id,
                                               results_root=results_root, asg_data_source=asg_data_source)
    semantic = None
    semantic_calls = []
    used_semantic = False
    label_grounding = None
    label_grounding_calls = []
    used_label_grounding = False
    progressive_outputs = {}
    direct = None
    direct_calls = []
    used_direct = False
    direct_first_prediction = None
    upstream_root = output / "scenes" / request.scene_id / "upstream"
    semantic_path = upstream_root / "semantic" / f"{request.task_id}.json"
    label_grounding_path = upstream_root / "label_grounding" / f"{request.task_id}.json"
    progressive_frontier_path = (
        upstream_root / "progressive_frontiers" / f"{request.task_id}.json"
    )
    direct_path = upstream_root / "direct" / f"{request.task_id}.json"
    direct_first_path = (
        upstream_root / "direct_first_prediction" / f"{request.task_id}.json"
    )

    # Generate all selected methods before consulting reference gold. With the
    # recommended method_workers=1, parallelism is across different tasks.
    pending = []
    for method in methods:
        target = target_for(method)
        if target.exists():
            record = read_json(target)
            if record.get("input_fingerprint") != fingerprint:
                raise ValueError(f"cached prediction input mismatch: {target}")
            continue
        pending.append((method, target))

    semantic_methods = {
        m("ReCoNav"),
        m("Forced-Early"),
        m("No-Verify-Repair"),
        m("Before-Repair"),
    }
    determine_label_grounding_methods = {
        m("ReCoNav"),
        m("Forced-Early"),
        m("No-Verify-Repair"),
        m("Before-Repair"),
    }
    progressive_methods = {m("No-Verify-Repair"), m("Before-Repair"), m("ReCoNav")}
    direct_methods = {m("Direct"), m("Direct-Feedback")}

    needs_semantic = [
        method for method, _ in pending if method in semantic_methods
    ]
    if needs_semantic:
        if semantic_path.exists():
            cached = read_json(semantic_path)
            if cached.get("fingerprint") != fingerprint:
                raise ValueError("cached semantic compilation does not match runtime input")
            semantic = _restore_semantic(cached)
            semantic_calls = cached["model_calls"]
            used_semantic = True
        else:
            try:
                semantic, semantic_calls = pipeline.compile_semantics(request)
            except Exception as exc:
                semantic = SemanticCompilationResult(
                    "", None, None,
                    infrastructure_error=f"{type(exc).__name__}: {exc}",
                )
                semantic_calls = []
            write_json(
                semantic_path,
                _compile_record(semantic, semantic_calls, fingerprint),
            )
    progressive_owner_candidate = next(
        (
            method
            for method in (m("ReCoNav"), m("Before-Repair"), m("No-Verify-Repair"))
            if method in needs_semantic
        ),
        None,
    )
    semantic_owner = (
        None
        if used_semantic or not needs_semantic
        else progressive_owner_candidate or needs_semantic[0]
    )

    needs_label_grounding = [
        method for method, _ in pending if method in determine_label_grounding_methods
    ]
    if needs_label_grounding:
        if label_grounding_path.exists():
            cached = read_json(label_grounding_path)
            if cached.get("fingerprint") != fingerprint:
                raise ValueError("cached label grounding artifact does not match runtime input")
            label_grounding = cached.get("artifact")
            label_grounding_calls = cached.get("model_calls", [])
            used_label_grounding = True
        else:
            contract = semantic.parsed_output if semantic is not None else None
            label_grounding, label_grounding_calls = pipeline.compile_label_grounding(
                request,
                contract,
                scene,
                grounding_policy="determine_first",
            )
            write_json(
                label_grounding_path,
                {
                    "fingerprint": fingerprint,
                    "grounding_policy": "determine_first",
                    "artifact": label_grounding,
                    "model_calls": label_grounding_calls,
                },
            )
    label_grounding_owner = (
        None
        if used_label_grounding or not needs_label_grounding
        else progressive_owner_candidate
        if progressive_owner_candidate in needs_label_grounding
        else needs_label_grounding[0]
    )

    progressive_pending = [
        method
        for method, _target in pending
        if method in progressive_methods
    ]
    progressive_charge_method = (
        m("ReCoNav")
        if m("ReCoNav") in progressive_pending
        else m("Before-Repair")
        if m("Before-Repair") in progressive_pending
        else m("No-Verify-Repair")
        if m("No-Verify-Repair") in progressive_pending
        else None
    )
    if progressive_pending:
        max_stage = (
            4
            if m("ReCoNav") in progressive_pending
            else 3
            if m("Before-Repair") in progressive_pending
            else 2
        )
        if progressive_frontier_path.exists():
            cached = read_json(progressive_frontier_path)
            if cached.get("fingerprint") != fingerprint:
                raise ValueError(
                    "cached progressive frontier does not match runtime input"
                )
            cached_stage = int(cached.get("max_stage", 0))
            if cached_stage < max_stage:
                raise ValueError(
                    "cached progressive frontier is shallower than requested stage"
                )
            raw_outputs = cached.get("predictions")
            if not isinstance(raw_outputs, dict):
                raise ValueError("cached progressive frontier is malformed")
            progressive_outputs = {
                method: _runtime_prediction_snapshot(value)
                for method, value in raw_outputs.items()
                if method in progressive_methods and isinstance(value, dict)
            }
        else:
            progressive_outputs = pipeline.run_progressive_frontiers(
                request,
                scene,
                semantic_result=semantic,
                semantic_calls=semantic_calls,
                semantic_reused=(
                    used_semantic or semantic_owner != progressive_charge_method
                ),
                label_grounding_result=label_grounding,
                label_grounding_calls=label_grounding_calls,
                label_grounding_reused=(
                    used_label_grounding or label_grounding_owner != progressive_charge_method
                ),
                provider=llm_provider,
                backbone=method_backbone(methods[0]),
                max_stage=max_stage,
            )
            write_json(
                progressive_frontier_path,
                {
                    "fingerprint": fingerprint,
                    "max_stage": max_stage,
                    "predictions": {
                        method: _runtime_prediction_snapshot(value)
                        for method, value in progressive_outputs.items()
                    },
                },
            )

    needs_direct = [
        method for method, _ in pending if method in direct_methods
    ]
    if needs_direct:
        if direct_path.exists():
            cached = read_json(direct_path)
            if cached.get("fingerprint") != fingerprint:
                raise ValueError("cached Direct first attempt does not match runtime input")
            direct = _restore_direct(cached)
            direct_calls = cached["model_calls"]
            used_direct = True
        else:
            direct, direct_calls = pipeline.compile_direct(request, scene)
            write_json(
                direct_path,
                _direct_record(direct, direct_calls, fingerprint),
            )
    direct_owner = None if used_direct or not needs_direct else needs_direct[0]

    if needs_direct:
        if direct_first_path.exists():
            cached = read_json(direct_first_path)
            if cached.get("fingerprint") != fingerprint:
                raise ValueError(
                    "cached complete Direct first result does not match runtime input"
                )
            direct_first_prediction = _runtime_prediction_snapshot(
                cached["prediction"]
            )
        else:
            direct_target = target_for(m("Direct"))
            if direct_target.exists():
                cached_direct_prediction = read_json(direct_target)
                if cached_direct_prediction.get("input_fingerprint") != fingerprint:
                    raise ValueError(
                        "cached Direct prediction does not match runtime input"
                    )
                direct_first_prediction = _runtime_prediction_snapshot(
                    cached_direct_prediction
                )
            else:
                direct_first_prediction = pipeline.run(
                    request,
                    scene,
                    m("Direct"),
                    direct_result=direct,
                    direct_calls=direct_calls,
                    direct_reused=used_direct,
                )
            write_json(
                direct_first_path,
                {
                    "fingerprint": fingerprint,
                    "prediction": _runtime_prediction_snapshot(
                        direct_first_prediction
                    ),
                },
            )

    def predict(method):
        if method == m("Direct"):
            if not isinstance(direct_first_prediction, dict):
                raise RuntimeError("complete Direct first result was not prepared")
            result = copy.deepcopy(direct_first_prediction)
            result["direct_pairing"] = {
                "complete_first_prediction_sha256": (
                    _runtime_prediction_digest(direct_first_prediction)
                )
            }
            return result

        if method in progressive_methods:
            result = progressive_outputs.get(method)
            if not isinstance(result, dict):
                raise RuntimeError(
                    f"progressive frontier {method} was not prepared"
                )
            return copy.deepcopy(result)

        semantic_reused = (
            method in semantic_methods and method != semantic_owner
        )
        direct_reused = (
            method in direct_methods and method != direct_owner
        )
        if method in determine_label_grounding_methods:
            method_label_grounding = label_grounding
            method_label_grounding_calls = label_grounding_calls
            method_label_grounding_reused = used_label_grounding or method != label_grounding_owner
        else:
            method_label_grounding = None
            method_label_grounding_calls = None
            method_label_grounding_reused = False
        result = pipeline.run(
            request,
            scene,
            method,
            semantic_result=semantic if method in semantic_methods else None,
            semantic_calls=semantic_calls if method in semantic_methods else None,
            semantic_reused=semantic_reused,
            label_grounding_result=method_label_grounding,
            label_grounding_calls=method_label_grounding_calls,
            label_grounding_reused=method_label_grounding_reused,
            direct_result=direct if method in direct_methods else None,
            direct_calls=direct_calls if method in direct_methods else None,
            direct_reused=direct_reused,
            direct_first_prediction=(
                direct_first_prediction
                if method == m("Direct-Feedback")
                else None
            ),
        )
        if (
            method == m("Direct-Feedback")
            and isinstance(direct_first_prediction, dict)
        ):
            result.setdefault("direct_feedback", {})[
                "shared_first_prediction_sha256"
            ] = _runtime_prediction_digest(direct_first_prediction)
        return result

    if not pending:
        predictions = []
    elif method_workers == 1:
        predictions = [(method, target, predict(method)) for method, target in pending]
    else:
        with ThreadPoolExecutor(max_workers=min(method_workers, len(pending))) as executor:
            futures = [
                (method, target, executor.submit(predict, method))
                for method, target in pending
            ]
            predictions = [
                (method, target, future.result())
                for method, target, future in futures
            ]

    statuses = {}
    for method, target, prediction in predictions:
        prediction.update(
            input_fingerprint=fingerprint,
            run_id=output.name,
            run_directory=str(output),
            result_method_id=result_method_id(method, asg_data_source),
            evaluation_status="pending",
            task_type=task.get("task_type"),
            difficulty=task.get("difficulty"),
        )
        write_json(target, prediction)
        status = (prediction.get("failure") or {}).get("reason", "completed")
        statuses[method] = status
        print(
            f"PREDICT {request.scene_id} {method} {request.task_id}: {status}",
            flush=True,
        )

    # Predictions/upstream responses are durable before stopping. A resume can
    # evaluate healthy saved predictions or archive/retry an interrupted pair.
    check_provider_stop(llm_provider)

    if os.environ.get('EXTERNAL_POLICY_DONE'):
        from reconav.baselines.common.time_budget import mark_policy_complete
        mark_policy_complete()

    reference_path = output / "scenes" / request.scene_id / "evaluation" / f"{request.task_id}.json"
    if prediction_only:
        reference = {"status": "prediction_only"}
    elif task_data_source != asg_data_source:
        reference = {
            "status": "requires_cross_source_alignment",
            "reason": "input and evaluation entity IDs are not assumed interchangeable",
        }
    elif reference_path.exists() and read_json(reference_path).get("status") == "valid":
        reference = read_json(reference_path)
        if reference.get("task_sha256") != task_hash:
            raise ValueError("reference cache input mismatch")
    else:
        reference = evaluate_reference(task, scene, reference_seconds)
    reference["task_sha256"] = task_hash
    write_json(reference_path, reference)

    for method in methods:
        target = target_for(method)
        record = read_json(target)
        if record.get("evaluation_status") == "evaluated":
            statuses.setdefault(
                method, (record.get("failure") or {}).get("reason", "completed")
            )
            continue
        can_score_route = (
            task_data_source == asg_data_source
            and reference["status"] in {"valid", "reference_unknown"}
        )
        if can_score_route:
            started = time.perf_counter()
            try:
                record.pop("evaluation_error", None)
                record["metrics"] = evaluate_prediction(
                    task,
                    record,
                    scene.asg,
                )
                record["evaluation_status"] = (
                    "evaluated"
                    if reference["status"] == "valid"
                    else "evaluated_partial_reference_unknown"
                )
            except Exception as exc:
                record["evaluation_status"] = "evaluator_error"
                record["evaluation_error"] = f"{type(exc).__name__}: {exc}"
            record.setdefault("timing", {})["evaluation_seconds"] = (
                time.perf_counter() - started
            )
        else:
            record["evaluation_status"] = reference["status"]
            record["metrics"] = {}
        write_json(target, record)
        statuses.setdefault(
            method, (record.get("failure") or {}).get("reason", "completed")
        )

    return {
        "scene_id": request.scene_id,
        "task_id": request.task_id,
        "methods": statuses,
        "reference_status": reference["status"],
    }


def run_experiment(*, tasks_root, asg_root, task_data_source, asg_data_source,
                   output, methods=None, config=None, scenes=(), task_ids=(), limit=None,
                   resume=False, model_info=None, client=None, prediction_only=False,
                   reference_seconds=90.0, pipeline_factory=None, method_workers=1,
                   task_workers=1, qwen_base_urls=(), task_order="mechanism-first", results_root=None):
    run_started = time.perf_counter()
    config = config or RuntimeConfig()
    output = Path(output).resolve()
    results_root = Path(results_root).resolve() if results_root is not None else None
    from reconav.benchmark.io import validate_run_id
    validate_run_id(output.name)
    tasks = select_tasks(tasks_root, scenes, task_ids, limit, task_order)
    methods = tuple(normalize_method(m) for m in (methods or methods_for_provider("qwen")))
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError("invalid method selection")
    llm_provider = method_provider(methods[0])
    if any(method_backbone(method) != method_backbone(methods[0]) for method in methods):
        raise ValueError(
            "one experiment run must use exactly one model backbone; "
            "cross-provider artifact sharing is forbidden"
        )
    qwen_endpoints: tuple[str, ...] = ()
    if llm_provider == "qwen":
        raw_endpoints = tuple(str(value).strip() for value in qwen_base_urls)
        if not raw_endpoints:
            model_endpoints = (
                model_info.get("endpoints")
                if isinstance(model_info, dict)
                else None
            )
            if isinstance(model_endpoints, list):
                raw_endpoints = tuple(str(value).strip() for value in model_endpoints)
        if not raw_endpoints:
            endpoint = (
                model_info.get("endpoint")
                if isinstance(model_info, dict)
                else None
            ) or os.environ.get(
                "QWEN_BASE_URL", "http://127.0.0.1:8008/v1"
            )
            raw_endpoints = (str(endpoint).strip(),)
        qwen_endpoints = tuple(dict.fromkeys(
            value.rstrip("/") for value in raw_endpoints if value
        ))
        if not qwen_endpoints:
            raise ValueError("Qwen run requires at least one base URL")
    if not isinstance(method_workers, int) or method_workers < 1:
        raise ValueError("method_workers must be a positive integer")
    if not isinstance(task_workers, int) or task_workers < 1:
        raise ValueError("task_workers must be a positive integer")
    if task_workers > 1 and (client is not None or pipeline_factory is not None):
        raise ValueError(
            "task-level process parallelism requires the production SharedLLMClient/"
            "NavigationPipeline; injected clients/factories must use task_workers=1"
        )

    code = repository_identity()
    model_info = model_info or {
        "provider": llm_provider if client is None else "injected_client",
        "served_model": "test_double" if client is not None else "unspecified",
    }
    if client is None and model_info.get("provider") != llm_provider:
        raise ValueError(
            "model_info provider does not match method provider"
        )
    settings = {
        "runtime_version": RUNTIME_VERSION,
        "provider": llm_provider,
        "backbone": method_backbone(methods[0]),
        "config": asdict(config),
        "methods": list(methods),
        "tasks_root": str(Path(tasks_root).resolve()),
        "asg_root": str(Path(asg_root).resolve()),
        "task_data_source": task_data_source,
        "asg_data_source": asg_data_source,
        "results_root": str(results_root) if results_root is not None else None,
        "model": model_info,
        "qwen_base_urls": list(qwen_endpoints),
        "source_hash": code["runtime_source_sha256"],
        "prompts": digest([
            load_system_prompt(),
            direct_prompt(),
            direct_feedback_prompt(),
            direct_id_sequence_prompt(),
            forced_early_binding_prompt(),
            final_fallback_binding_prompt(),
        ]),
        "prediction_only": prediction_only,
        "reference_seconds": reference_seconds,
        "method_workers": method_workers,
        "task_workers": task_workers,
        "task_order": task_order,
        "parallelism": (
            "task_processes"
            if task_workers > 1
            else "single_task_process"
        ),
    }
    config_hash = digest(settings)
    manifest_path = output / "run_manifest.json"
    if output.exists() and not resume:
        raise FileExistsError(
            f"output already exists; use --resume for identical settings: {output}"
        )
    if resume and manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["configuration_fingerprint"] != config_hash:
            raise ValueError(
                "resume settings/code/model differ; use a new run directory, do not mix predictions"
            )
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("nonempty output has no valid run manifest")
        manifest = {
            "configuration_fingerprint": config_hash,
            "settings": settings,
            "repository": code,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "tasks": {},
            "status": "running",
            "gold_boundary": (
                "NavigationRequest whitelist only; reference solving and evaluation "
                "occur after predictions."
            ),
        }
    manifest["status"] = "running"
    output.mkdir(parents=True, exist_ok=True)

    # Lock the run once in the parent process. Child task workers only touch
    # disjoint per-task paths; manifest/summary remain parent-owned.
    lock = output / ".writer.lock"
    import fcntl
    with lock.open("a+") as lock_file, ExitStack() as lifecycle:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another process is writing this run") from exc
        lifecycle.enter_context(provider_run_guard(output, resume=resume))

        if results_root is not None:
            # Public method directories belong to one run. Separate backbone
            # locks allow the four models to write concurrently without mixing
            # a second batch into these paper result directories.
            owners = results_root / "_runs" / "owners"
            owners.mkdir(parents=True, exist_ok=True)
            claims = []
            for method in sorted(methods):
                identifier = result_method_id(method, asg_data_source)
                handle = lifecycle.enter_context((owners / f"{identifier}.lock").open("a+"))
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError(f"another process is writing {identifier}") from exc
                owner_path = owners / f"{identifier}.json"
                owner = {"run_directory": str(output), "configuration_fingerprint": config_hash,
                         "method": method, "result_method_id": identifier}
                if owner_path.exists():
                    if read_json(owner_path) != owner:
                        raise ValueError(f"result directory belongs to a different run: {identifier}; use a new results root")
                elif (results_root / identifier).exists() and any((results_root / identifier).iterdir()):
                    raise ValueError(f"unowned result directory is not empty: {identifier}")
                claims.append((owner_path, owner))
            for owner_path, owner in claims:
                write_json(owner_path, owner)

        provider = DirectorySceneProvider(Path(asg_root), asg_data_source)
        jobs = []
        for task_ordinal, (path, task, request) in enumerate(tasks):
            task_key = f"{request.scene_id}/{request.task_id}"
            task_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if (
                task_key in manifest["tasks"]
                and manifest["tasks"][task_key]["sha256"] != task_hash
            ):
                raise ValueError(f"task changed under existing run ID: {task_key}")
            scene = provider.get(request.scene_id)
            # Validate/cache the public representation before launching workers;
            # no task-conditioned cropping is introduced by parallel scheduling.
            scene.public_view(config.scene_geometry)
            previous = manifest["tasks"].get(task_key)
            if previous and previous["asg_fingerprint"] != scene.fingerprint:
                raise ValueError(
                    f"ASG changed under existing run ID: {request.scene_id}"
                )
            manifest["tasks"][task_key] = {
                "path": str(path),
                "sha256": task_hash,
                "asg_fingerprint": scene.fingerprint,
            }
            jobs.append({
                "path": str(path),
                "task": task,
                "request": request,
                "qwen_base_url": (
                    qwen_endpoints[task_ordinal % len(qwen_endpoints)]
                    if qwen_endpoints
                    else None
                ),
                "expected_scene_fingerprint": scene.fingerprint,
                "output": str(output),
                "results_root": str(results_root) if results_root is not None else None,
                "methods": tuple(methods),
                "config_hash": config_hash,
                "task_data_source": task_data_source,
                "asg_data_source": asg_data_source,
                "prediction_only": prediction_only,
                "reference_seconds": reference_seconds,
                "method_workers": method_workers,
            })
        write_json(manifest_path, manifest)

        interruption = None
        try:
            if task_workers == 1:
                pipeline = (pipeline_factory or NavigationPipeline)(
                    client or SharedLLMClient(llm_provider), config
                )
                for completed, job in enumerate(jobs, 1):
                    task_job = dict(job)
                    qwen_base_url = task_job.pop("qwen_base_url", None)
                    if qwen_base_url is not None:
                        os.environ["QWEN_BASE_URL"] = str(qwen_base_url)
                    result = _run_one_task(provider=provider, pipeline=pipeline, **task_job)
                    print(
                        f"TASK [{completed}/{len(jobs)}] "
                        f"{result['scene_id']} {result['task_id']} "
                        f"reference={result['reference_status']}", flush=True,
                    )
            else:
                # Bound the pending queue too: quota exhaustion must not submit
                # the remaining dataset. Active workers share the stop marker;
                # already-sent requests may finish and save their records.
                ctx = multiprocessing.get_context("spawn")
                workers = min(task_workers, len(jobs))
                with ProcessPoolExecutor(
                    max_workers=workers, mp_context=ctx,
                    initializer=_initialize_task_process,
                    initargs=(str(Path(asg_root).resolve()), asg_data_source, config, llm_provider),
                ) as executor:
                    remaining = iter(jobs)
                    futures = {executor.submit(_run_task_process, next(remaining)) for _ in range(workers)}
                    completed = 0
                    try:
                        while futures:
                            done, _ = wait(futures, return_when=FIRST_COMPLETED)
                            for future in done:
                                futures.remove(future)
                                result = future.result()
                                completed += 1
                                print(
                                    f"TASK [{completed}/{len(jobs)}] "
                                    f"{result['scene_id']} {result['task_id']} "
                                    f"reference={result['reference_status']}", flush=True,
                                )
                                check_provider_stop(llm_provider)
                                job = next(remaining, None)
                                if job is not None:
                                    futures.add(executor.submit(_run_task_process, job))
                    finally:
                        for future in futures:
                            future.cancel()
        except ProviderRunStopped as exc:
            interruption = stop_record() or {"provider": llm_provider, "reason": str(exc)}
            print(f"STOPPED {exc}; saved outputs remain in {output}", flush=True)

        summary = summarize_run(output, methods)
        if interruption:
            summary["provider_interruption"] = interruption
            summary["valid_for_reporting"] = False
            write_json(output / "summary.json", summary)
            manifest["provider_interruption"] = interruption
        else:
            manifest.pop("provider_interruption", None)
        manifest["status"] = ("stopped_provider_quota" if interruption else
                              "complete" if summary["valid_for_reporting"] else "incomplete")
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["execution_seconds"] = float(manifest.get("execution_seconds", 0.0)) + time.perf_counter() - run_started
        write_json(manifest_path, manifest)

    return summary
