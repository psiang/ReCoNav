"""Method cost and run health, including shared upstream work."""
from __future__ import annotations


def logical_model_calls(prediction):
    upstream = prediction.get("upstream") or {}
    calls = [*prediction.get("model_calls", []),
             *upstream.get("semantic_calls", []), *upstream.get("label_grounding_calls", []),
             *(prediction.get("direct_source") or {}).get("model_calls", [])]
    seen, result = set(), []
    for call in calls:
        identifier = call.get("call_id")
        if identifier and identifier in seen:
            continue
        if identifier:
            seen.add(identifier)
        result.append(call)
    return result


def infrastructure_errors(prediction):
    errors = [str(call["error"]) for call in logical_model_calls(prediction) if call.get("error")]
    for name in ("semantic_source", "direct_source"):
        source = prediction.get(name) or {}
        if source.get("infrastructure_error"):
            errors.append(str(source["infrastructure_error"]))
    failure = prediction.get("failure") or {}
    if "INFRASTRUCTURE" in str(failure.get("reason", "")):
        errors.append(str(failure))
    return errors


def complete_accounting(prediction, *, inline_upstream_seconds=0.0):
    timing = prediction.setdefault("timing", {})
    upstream = prediction.get("upstream") or {}
    direct = prediction.get("direct_source") or {}
    upstream_seconds = float((upstream.get("semantic_timing") or {}).get("total_seconds", 0.0))
    upstream_seconds += sum(float(call.get("seconds", 0.0)) for call in upstream.get("label_grounding_calls", []))
    upstream_seconds += float((direct.get("timing") or {}).get("total_seconds", 0.0))
    execution = timing.get("execution_pipeline_seconds", timing.get("pipeline_seconds", 0.0) - inline_upstream_seconds)
    timing.update(execution_pipeline_seconds=max(0.0, execution),
                  upstream_seconds=upstream_seconds,
                  pipeline_seconds=max(0.0, execution) + upstream_seconds)
    calls = logical_model_calls(prediction)
    prediction["logical_cost"] = {
        "model_calls": sum(bool(call.get("provider_called")) for call in calls),
        "llm_seconds": sum(float(call.get("seconds", 0.0)) for call in calls),
        **{key: sum(float((call.get("usage") or {}).get(key, 0)) for call in calls)
           for key in ("input_tokens", "output_tokens", "billable_output_tokens", "total_tokens", "cached_input_tokens", "cache_write_input_tokens", "reasoning_tokens")},
        "solver_queries": len(prediction.get("solver_queries", [])),
        "solver_seconds": timing.get("solver_seconds", 0.0),
    }
    prediction["infrastructure_errors"] = infrastructure_errors(prediction)
    return prediction
