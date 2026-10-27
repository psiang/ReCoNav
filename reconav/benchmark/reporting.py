"""Collect new experiment records and report the paper's evaluation views."""
from __future__ import annotations

from collections import Counter
import csv
from importlib.resources import files
import json
import math
from pathlib import Path

from reconav.benchmark.accounting import aggregate, fixed_quality, scoring_complete
from reconav.benchmark.selection import task_ids


METRICS = ("full_is", "hard_ns", "prefer", "seconds", "total_tokens")
TERMINAL = {"completed", "timeout", "resource_exhausted", "unsupported_input", "failed"}
EXTERNAL_METHOD_NAMES = {
    "navgpt": "NavGPT", "mapgpt": "MapGPT", "canav": "CA-Nav",
    "gcvln": "GC-VLN", "snav": "SNav", "dualvln": "DualVLN",
    "lang2ltl2": "Lang2LTL-2", "limp": "LIMP",
}


def read(path):
    return json.loads(Path(path).read_text())


def external_method_name(method, condition):
    return EXTERNAL_METHOD_NAMES[method] + ("+ASG" if condition == "asg" else "")


def internal_rows(run):
    from reconav.baselines.common.time_budget import internal_resource_reason
    from reconav.method.runtime.accounting import infrastructure_errors
    from reconav.method.runtime.models import method_backbone, method_variant, paper_method_name
    from reconav.method.runtime.result_layout import prediction_paths

    manifest = read(run / "run_manifest.json")
    settings = manifest["settings"]
    expected = {k.rsplit("/", 1)[-1] for k in manifest["tasks"]}
    for method in settings["methods"]:
        model, variant = method_backbone(method), paper_method_name(method_variant(method))
        rows = []
        for path in prediction_paths(run, method, manifest):
            record = read(path)
            if record["task_id"] not in expected or record["method"] != method:
                raise ValueError(f"Unexpected prediction: {path}")
            reason = internal_resource_reason(record)
            error = infrastructure_errors(record) or record.get("infrastructure_errors")
            scored = record.get("evaluation_status") == "evaluated"
            status = "failed" if error or not scored else "resource_exhausted" if reason else "completed"
            metrics = record.get("metrics") or {}
            rows.append(dict(task_id=record["task_id"], execution_status=status,
                             scoring_completed=scored, resource_reason=reason,
                             full_is=metrics.get("full_is"), hard_ns=metrics.get("hard_ns"),
                             prefer=metrics.get("preference_satisfaction"),
                             seconds=record.get("timing", {}).get("pipeline_seconds"),
                             total_tokens=record.get("logical_cost", {}).get("total_tokens"),
                             error=error or record.get("evaluation_error"),
                             source=str(path)))
        yield ("internal", model, variant, "asg"), expected, rows


def external_rows(run):
    from reconav.baselines.common.time_budget import resource_reason

    plan = read(run / "evaluation_plan.json")["entries"]
    plans, actual = {}, {}
    for row in plan:
        key = ("external", "qwen" if row["method"] not in {"snav", "dualvln"} else "checkpoint",
               external_method_name(row["method"], row.get("condition", "native")), row.get("condition", "native"))
        plans.setdefault(key, set()).add(row["task_id"])
    for path in sorted(run.rglob("status.json")):
        status = read(path)
        if path.parent.name != status.get("task_id"):
            continue
        key = ("external", "qwen" if status["method"] not in {"snav", "dualvln"} else "checkpoint",
               external_method_name(status["method"], status.get("condition", "native")), status.get("condition", "native"))
        if key not in plans or status["task_id"] not in plans[key]:
            raise ValueError(f"Status not declared in evaluation plan: {path}")
        metrics_path = path.parent / "metrics.json"
        metrics = read(metrics_path) if metrics_path.is_file() else {}
        row = {k: status.get(k) for k in ("task_id", "execution_status", "scoring_completed", "seconds", "total_tokens", "error", "time_protocol")}
        row.update({k: metrics.get(k) for k in ("full_is", "hard_ns", "prefer")})
        row["resource_reason"] = resource_reason(status, path.parent)
        if row["resource_reason"] == "dollar_budget_exhausted":
            row["execution_status"] = "failed"
        if status.get("token_usage_complete") is False:
            row["total_tokens"] = None
        if "total_tokens" not in status and (path.parent / "calls.json").exists():
            calls = read(path.parent / "calls.json")
            row["total_tokens"] = (sum(c["usage"]["total_tokens"] for c in calls)
                                   if all("total_tokens" in (c.get("usage") or {}) for c in calls)
                                   and status.get("token_usage_complete") is not False else None)
        row["source"] = str(path)
        actual.setdefault(key, []).append(row)
    for key, expected in plans.items():
        yield key, expected, actual.get(key, [])


def check_row(row, preferences):
    if row["execution_status"] not in TERMINAL:
        return row["execution_status"]
    if row["execution_status"] == "unsupported_input":
        return "unsupported task in supported population"
    if row["execution_status"] == "failed":
        return None
    value = row.get("seconds")
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return "missing or invalid runtime"
    if row["execution_status"] == "completed" and not row.get("scoring_completed"):
        return "scoring incomplete"
    if scoring_complete(row):
        for key in ("full_is", "hard_ns", *(["prefer"] if row["task_id"] in preferences else [])):
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                return "missing or invalid " + key
    return None


def report_view(rows, population, supported, preferences, cap=None):
    selected = []
    for source in rows:
        if source["task_id"] not in population:
            continue
        row = dict(source)
        seconds = row.get("seconds")
        # An infrastructure error remains an error even if it took over 300 s.
        if cap is not None and row["execution_status"] in {"completed", "timeout", "resource_exhausted"} and isinstance(seconds, (int, float)) and seconds > cap:
            row.update(execution_status="timeout", scoring_completed=False,
                       resource_reason=f"uniform_{cap:g}s_retrospective_censoring")
        selected.append(row)
    by_id = {r["task_id"]: r for r in selected}
    missing = sorted(supported - by_id.keys())
    issues = []
    for task in sorted(supported & by_id.keys()):
        row = by_id[task]
        reason = check_row(row, preferences)
        policy = row.get("time_protocol") or {}
        limit = policy.get("limit_seconds")
        if row["execution_status"] == "timeout" and (row.get("resource_reason") or "").startswith("uniform_"):
            pass
        elif row["execution_status"] == "timeout" and limit and (cap is None or limit < cap):
            reason = f"source wall limit {limit:g}s is shorter than this report's horizon"
        if reason:
            issues.append(dict(task_id=task, reason=reason))
    summary = aggregate(selected, len(population), scoring_complete)
    valid = not missing and not issues
    quality = fixed_quality(selected, supported, preferences) if valid else None
    summary.update(ready=valid, supported=len(supported), missing_tasks=missing,
                   issues=issues, statuses=dict(Counter(r["execution_status"] for r in selected)),
                   timeout_rate=sum(r["execution_status"] == "timeout" for r in selected) / len(population),
                   quality_denominator=len(supported), prefer_denominator=len(supported & preferences),
                   fixed_quality=quality["quality"] if quality else None)
    return summary


def summarize(data, runs, output, split="all", requested=()):
    tasks = {p.stem: read(p) for p in (data / "benchmark/tasks/hm3dsem").glob("*/*.json")}
    all_ids = set(read(data / "benchmark/splits/all.json"))
    if set(tasks) != all_ids:
        raise ValueError("Task files and benchmark split disagree")
    selected = set(task_ids(data / "benchmark/tasks/hm3dsem", split, requested)) or all_ids
    if not selected <= all_ids:
        raise ValueError("Unknown task IDs: " + str(sorted(selected - all_ids)))
    native = set(json.loads(files("reconav.benchmark").joinpath("protocols/native_supported.json").read_text()))
    hard = set(read(data / "benchmark/splits/hard44.json"))
    preferences = {t for t, task in tasks.items() if any(c["kind"].startswith("prefer") for c in task["symbolic_ir"]["constraints"])}
    groups = {}
    for run in runs:
        run = run.expanduser().resolve()
        loader = internal_rows if (run / "run_manifest.json").is_file() else external_rows
        if loader is external_rows and not (run / "evaluation_plan.json").is_file():
            raise ValueError(f"No experiment manifest in {run}")
        for key, expected, rows in loader(run):
            group = groups.setdefault(key, dict(planned=set(), rows={}))
            overlap = group["planned"] & expected
            if overlap:
                raise ValueError(f"Duplicate task/method input {key}: {sorted(overlap)[:3]}")
            group["planned"].update(expected)
            for row in rows:
                if row["task_id"] in group["rows"]:
                    raise ValueError(f"Duplicate record in {run}: {row['task_id']}")
                group["rows"][row["task_id"]] = row
    if not groups:
        raise ValueError("No methods declared in the supplied runs")
    reports, flat = [], []
    for (kind, model, method, condition), group in sorted(groups.items()):
        label = dict(kind=kind, model=model, method=method, condition=condition)
        rows = list(group["rows"].values())
        flat.extend(dict(**label, **r) for r in rows if r["task_id"] in selected)
        supported = selected & (native if condition == "native" else all_ids)
        views = [("uncapped", selected, None), ("main_300s", selected, 300), ("hard44_uncapped", selected & hard, None)]
        for view, population, cap in views:
            if not population:
                continue
            result = report_view(rows, population, supported & population, preferences, cap)
            reports.append(dict(**label, view=view, **result))
    paired = []
    for kind, model, method, condition in sorted(groups):
        if kind != "internal" or method != "ReCoNav":
            continue
        early = groups.get((kind, model, "Forced-Early", condition))
        if early is None:
            continue
        reconav = groups[kind, model, method, condition]
        resolving = {t for t, task in tasks.items()
                     if task["uniqueness_proof"]["identity_class"] in {"hard_constraint", "soft_preference"}}
        for name, members in (("resolving89", resolving), ("control103", all_ids - resolving)):
            population = selected & members
            if not population:
                continue
            pairs = [t for t in sorted(population) if t in early["rows"] and t in reconav["rows"]
                     and all(not check_row(g["rows"][t], preferences) and scoring_complete(g["rows"][t]) for g in (early, reconav))]
            paired.append(dict(model=model, group=name, planned=len(population), paired_completed=len(pairs),
                               forced_early_successes=sum(early["rows"][t]["full_is"] for t in pairs),
                               reconav_successes=sum(reconav["rows"][t]["full_is"] for t in pairs)))
    output.mkdir(parents=True, exist_ok=True)
    document = dict(task_ids=sorted(selected), reports=reports, paired=paired)
    (output / "summary.json").write_text(json.dumps(document, indent=2) + "\n")
    (output / "rows.json").write_text(json.dumps(sorted(flat, key=lambda r: (r["kind"], r["model"], r["method"], r["condition"], r["task_id"])), indent=2) + "\n")
    columns = ["kind", "model", "method", "condition", "view", "ready", "planned", "observed", "supported", "completed", "quality_denominator", "prefer_denominator", "completion_rate", "timeout_rate"]
    columns += [f"{scope}_{key}" for scope in ("fixed", "completed") for key in METRICS]
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for r in reports:
            row = {k: r.get(k) for k in columns}
            for key in METRICS:
                row["fixed_" + key] = (r.get("fixed_quality") or {}).get(key, {}).get("mean")
                row["completed_" + key] = r["means"][key]["mean"]
            writer.writerow(row)
    with (output / "paired.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "group", "planned", "paired_completed", "forced_early_successes", "reconav_successes"])
        writer.writeheader()
        writer.writerows(paired)
    lines = ["# Experiment results", "", f"Selected tasks: {len(selected)}.", "",
             "| View | Model / method | Ready | Completed / planned | Full IS (%) | Hard NS (%) | Prefer (%) | Tokens (k) | Time (s) |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    def fmt(v, scale=1):
        return "—" if v is None else f"{v * scale:.2f}"
    for r in reports:
        scores = r.get("fixed_quality") or {}
        if r["view"] == "uncapped" and r["kind"] == "internal":
            scores = r["means"]
        lines.append("| " + " | ".join([r["view"], f"{r['model']} / {r['method']} ({r['condition']})",
                     "yes" if r["ready"] else "pending", f"{r['completed']} / {r['planned']}",
                     *[fmt(scores.get(k, {}).get("mean"), 100) for k in ("full_is", "hard_ns", "prefer")],
                     fmt(r["means"]["total_tokens"]["mean"], .001), fmt(r["means"]["seconds"]["mean"])]) + " |")
    for r in reports:
        if r["missing_tasks"] or r["issues"]:
            lines.extend(["", f"{r['view']} / {r['model']} / {r['method']} ({r['condition']}): "
                          f"{len(r['missing_tasks'])} missing, {len(r['issues'])} errors. Details: summary.json."])
    if paired:
        lines.extend(["", "## Forced-Early → ReCoNav", "", "| Model | Group | Paired / planned | Forced-Early successes | ReCoNav successes |", "|---|---|---:|---:|---:|"])
        lines.extend(f"| {r['model']} | {r['group']} | {r['paired_completed']} / {r['planned']} | {r['forced_early_successes']} | {r['reconav_successes']} |" for r in paired)
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return dict(methods=len(groups), views=len(reports), selected_tasks=len(selected),
                ready_views=sum(r["ready"] for r in reports), output=str(output))
