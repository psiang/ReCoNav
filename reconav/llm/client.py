"""Shared Qwen, OpenAI, and Gemini language-model client with per-thread usage."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from reconav.llm.control import check_provider_stop, stop_on_quota

_QWEN_BASE_URL = "http://127.0.0.1:8008/v1"
_QWEN_MODEL = "qwen"
_OPENAI_BASE_URL = "https://api.openai.com/v1"
_OPENAI_MODEL = "gpt-5.6-sol"
_OPENAI_REASONING = "high"
_OPENAI_REASONING_TOKEN_RESERVE = 8192
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
LLM_SEED = int(os.environ.get("LLM_SEED", "0"))
_THREAD_USAGE = threading.local()


def current_model_config(provider: str | None = None) -> dict[str, Any]:
    provider = (provider or os.environ.get("LLM_PROVIDER", "qwen")).strip().lower()
    if provider == "gpt":
        provider = "openai"
    if provider == "qwen":
        return {
            "provider": "qwen",
            "model": os.environ.get("QWEN_SERVED_MODEL_NAME", _QWEN_MODEL),
            "api": "chat.completions",
            "reasoning_effort": None,
            "seed": LLM_SEED,
            "endpoint": os.environ.get("QWEN_BASE_URL", _QWEN_BASE_URL),
        }
    if provider == "openai":
        model = os.environ.get("OPENAI_MODEL", _OPENAI_MODEL)
        reasoning = os.environ.get("OPENAI_REASONING_EFFORT", _OPENAI_REASONING)
        if model != _OPENAI_MODEL or reasoning != _OPENAI_REASONING:
            raise ValueError(
                f"closed-model protocol requires {_OPENAI_MODEL} with reasoning {_OPENAI_REASONING}"
            )
        return {
            "provider": "openai",
            "model": model,
            "api": "responses",
            "reasoning_effort": reasoning,
            "seed": None,
        }
    if provider == "gemini":
        model = os.environ.get("GEMINI_MODEL")
        if not model:
            raise ValueError("GEMINI_MODEL is required for the gemini provider")
        return {
            "provider": "gemini",
            "model": model,
            "api": "streamGenerateContent",
            "reasoning_effort": None,
            "seed": LLM_SEED,
        }
    raise ValueError(f"unsupported LLM_PROVIDER: {provider!r}")


def usage_snapshot() -> dict[str, int | float]:
    stats = getattr(_THREAD_USAGE, "stats", None)
    if stats is None:
        stats = {
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "llm_latency_seconds": 0.0,
            "qwen_calls": 0,
            "qwen_input_tokens": 0,
            "qwen_output_tokens": 0,
            "qwen_total_tokens": 0,
            "qwen_latency_seconds": 0.0,
            "openai_calls": 0,
            "openai_input_tokens": 0,
            "openai_output_tokens": 0,
            "openai_total_tokens": 0,
            "openai_latency_seconds": 0.0,
            "gemini_calls": 0,
            "gemini_input_tokens": 0,
            "gemini_output_tokens": 0,
            "gemini_total_tokens": 0,
            "gemini_latency_seconds": 0.0,
        }
        _THREAD_USAGE.stats = stats
    return dict(stats)


def usage_since(snapshot: dict[str, int | float]) -> dict[str, int | float]:
    current = usage_snapshot()
    return {key: current[key] - snapshot.get(key, 0) for key in current}


def transport_attempt_snapshot(provider: str) -> int:
    provider = str(provider).strip().lower()
    attempts = getattr(_THREAD_USAGE, "transport_attempts", None)
    if attempts is None:
        attempts = {}
        _THREAD_USAGE.transport_attempts = attempts
    rows = attempts.setdefault(provider, [])
    return len(rows)


def transport_attempts_since(provider: str, snapshot: int) -> list[dict[str, Any]]:
    provider = str(provider).strip().lower()
    transport_attempt_snapshot(provider)
    return [
        dict(item)
        for item in _THREAD_USAGE.transport_attempts[provider][snapshot:]
    ]


def _record_transport_attempt(
    provider: str,
    attempt: int,
    status: str,
    *,
    timed_out: bool,
    endpoint: str | None = None,
) -> None:
    provider = str(provider).strip().lower()
    transport_attempt_snapshot(provider)
    record = {
        "provider": provider,
        "attempt": attempt,
        "status": status,
        "timed_out": timed_out,
    }
    if endpoint is not None:
        record["endpoint"] = str(endpoint)
    _THREAD_USAGE.transport_attempts[provider].append(record)


def chat_completion(
    *,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float = 0,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
    usage: dict[str, int] | None = None,
    seed: int | None = LLM_SEED,
    enable_thinking: bool = False,
    response_format: dict[str, Any] | None = None,
    provider: str | None = None,
) -> str:
    config = current_model_config(provider)
    started = time.perf_counter()
    if config["provider"] == "qwen":
        data = _qwen_request(
            messages,
            max_tokens,
            temperature,
            top_p,
            top_k,
            min_p,
            presence_penalty,
            repetition_penalty,
            seed,
            enable_thinking,
            response_format,
        )
        response_usage = data.get("usage", {})
        input_tokens = int(response_usage.get("prompt_tokens", 0))
        output_tokens = int(response_usage.get("completion_tokens", 0))
        total_tokens = int(response_usage.get("total_tokens", input_tokens + output_tokens))
    elif config["provider"] == "openai":
        data = _openai_request(messages, max_tokens, response_format, config)
        response_usage = data.get("usage", {})
        input_tokens = int(response_usage.get("input_tokens", 0))
        output_tokens = int(response_usage.get("output_tokens", 0))
        total_tokens = int(response_usage.get("total_tokens", input_tokens + output_tokens))
    else:
        data = _gemini_request(
            messages,
            max_tokens,
            temperature,
            top_p,
            top_k,
            seed,
            response_format,
            config,
        )
        response_usage = data.get("usage", {})
        input_tokens = int(response_usage.get("total_input_tokens", 0))
        output_tokens = int(response_usage.get("total_output_tokens", 0))
        total_tokens = int(
            response_usage.get("total_tokens", input_tokens + output_tokens)
        )
    elapsed = time.perf_counter() - started
    _record_usage(
        str(config["provider"]),
        input_tokens,
        output_tokens,
        total_tokens,
        elapsed,
        usage,
    )
    details = {
        "cache_write_input_tokens": int(
            (response_usage.get("input_tokens_details") or {}).get("cache_write_tokens", 0)
            if config["provider"] == "openai" else 0
        ),
        "cached_input_tokens": int(
            response_usage.get("input_tokens_details", {}).get("cached_tokens", 0)
            if config["provider"] == "openai" else
            response_usage.get("total_cached_tokens", 0) if config["provider"] == "gemini" else
            (response_usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        ),
        "reasoning_tokens": int(
            response_usage.get("output_tokens_details", {}).get("reasoning_tokens", 0)
            if config["provider"] == "openai" else
            response_usage.get("total_thought_tokens", 0) if config["provider"] == "gemini" else
            (response_usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        ),
    }
    stats = usage_snapshot()
    for key, value in details.items():
        stats[key] = stats.get(key, 0) + value
    # Gemini reports thought tokens separately from visible output; both are
    # billable output. Keep visible output and total usage unmodified.
    stats["billable_output_tokens"] = stats.get("billable_output_tokens", 0) + output_tokens + (
        details["reasoning_tokens"] if config["provider"] == "gemini" else 0
    )
    _THREAD_USAGE.stats = stats
    _THREAD_USAGE.response_metadata = {
        "usage": response_usage, "model": data.get("model"),
        "status": data.get("status"), "incomplete_details": data.get("incomplete_details"),
    }
    if config["provider"] == "gemini":
        _THREAD_USAGE.response_metadata.update(data.get("transport", {}))
    if config["provider"] == "qwen":
        choice = data["choices"][0]
        _THREAD_USAGE.response_metadata["finish_reason"] = choice.get("finish_reason")
        # A returned, budget-limited completion is a model outcome, not a
        # transport outage. Preserve its text for the common parser and for
        # feedback; never resample a whole task because the model ran long.
        return str(choice["message"].get("content") or "")
    if config["provider"] == "openai":
        return _response_text(data)
    return _gemini_response_text(data)


def response_metadata_snapshot():
    return dict(getattr(_THREAD_USAGE, "response_metadata", {}))


def request_metadata_snapshot():
    return dict(getattr(_THREAD_USAGE, "request_metadata", {}))


def _qwen_request(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    min_p: float | None,
    presence_penalty: float | None,
    repetition_penalty: float | None,
    seed: int | None,
    enable_thinking: bool,
    response_format: dict[str, Any] | None,
) -> dict[str, Any]:
    base_url = os.environ.get("QWEN_BASE_URL", _QWEN_BASE_URL)
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("QWEN_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    timeout = float(os.environ.get("QWEN_TIMEOUT_S", "240"))
    retries = max(1, int(os.environ.get("QWEN_RETRIES", "2")))
    request_max_tokens = int(max_tokens)
    last_error: Exception | None = None
    for attempt in range(retries):
        payload: dict[str, Any] = {
            "model": os.environ.get("QWEN_SERVED_MODEL_NAME", _QWEN_MODEL),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": request_max_tokens,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        for key, value in {
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
        }.items():
            if value is not None:
                payload[key] = value
        if seed is not None:
            payload["seed"] = int(seed)
        if response_format is not None:
            payload["response_format"] = response_format
        try:
            response = _post_json(url, payload, headers, timeout)
        except HTTPError as exc:
            _record_qwen_attempt(
                attempt + 1,
                "http_error",
                timed_out=False,
                endpoint=base_url,
            )
            detail = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code in {408, 409, 429} or 500 <= exc.code < 600
            if not retryable or attempt + 1 >= retries:
                raise RuntimeError(
                    f"Qwen request failed with HTTP {exc.code}: {detail}"
                ) from exc
            last_error = exc
            time.sleep(min(8.0, 1.0 + attempt))
        except (URLError, TimeoutError, socket.timeout, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            timed_out = _is_timeout_error(exc)
            _record_qwen_attempt(
                attempt + 1,
                "timeout" if timed_out else "transport_error",
                timed_out=timed_out,
                endpoint=base_url,
            )
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(1.0 + attempt)
        else:
            _record_qwen_attempt(
                attempt + 1,
                "success",
                timed_out=False,
                endpoint=base_url,
            )
            return response
    raise RuntimeError(f"Qwen request failed to reach {url}: {last_error}") from last_error


def _record_qwen_attempt(
    attempt: int,
    status: str,
    *,
    timed_out: bool,
    endpoint: str | None = None,
) -> None:
    _record_transport_attempt(
        "qwen",
        attempt,
        status,
        timed_out=timed_out,
        endpoint=endpoint,
    )


def _is_timeout_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout, subprocess.TimeoutExpired)):
        return True
    return isinstance(exc, URLError) and isinstance(
        exc.reason,
        (TimeoutError, socket.timeout),
    )


def _closed_provider_post(
    *,
    provider: str,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    """Retry the exact same remote request only for transport/transient failures."""
    retries = max(1, int(retries))
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        check_provider_stop(provider)
        try:
            data = _post_json(url, payload, headers, timeout)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code in {408, 409, 429} or 500 <= exc.code < 600
            _record_transport_attempt(
                provider,
                attempt,
                "http_error",
                timed_out=False,
            )
            stop_on_quota(provider, code=exc.code, detail=detail)
            last_error = RuntimeError(
                f"{provider} request failed with HTTP {exc.code}: {detail}"
            )
            if not retryable or attempt >= retries:
                raise last_error from exc
        except (
            URLError,
            TimeoutError,
            socket.timeout,
            subprocess.SubprocessError,
            json.JSONDecodeError,
        ) as exc:
            timed_out = _is_timeout_error(exc)
            _record_transport_attempt(
                provider,
                attempt,
                "timeout" if timed_out else "transport_error",
                timed_out=timed_out,
            )
            last_error = exc
            if attempt >= retries:
                raise
        else:
            _record_transport_attempt(
                provider,
                attempt,
                "success",
                timed_out=False,
            )
            return data
        time.sleep(min(8.0, 1.0 * (2 ** (attempt - 1))))
    raise RuntimeError(
        f"{provider} request failed after {retries} attempts: {last_error}"
    ) from last_error


def _openai_request(
    messages: list[dict[str, str]],
    max_tokens: int,
    response_format: dict[str, Any] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for the openai provider")
    url = os.environ.get("OPENAI_BASE_URL", _OPENAI_BASE_URL).rstrip("/") + "/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": config["model"],
        "input": messages,
        "reasoning": {"effort": config["reasoning_effort"]},
        # Responses counts reasoning tokens against this limit; callers specify
        # the visible-answer budget used by the rest of the pipeline.
        "max_output_tokens": int(max_tokens) + _OPENAI_REASONING_TOKEN_RESERVE,
    }
    if response_format is not None:
        payload["text"] = {
            "format": _openai_text_format(response_format)
        }
    timeout = float(os.environ.get("OPENAI_TIMEOUT_S", "300"))
    retries = int(os.environ.get("OPENAI_RETRIES", "3"))
    return _closed_provider_post(
        provider="openai",
        url=url,
        payload=payload,
        headers=headers,
        timeout=timeout,
        retries=retries,
    )


def _schema_components(response_format: dict[str, Any]) -> dict[str, Any]:
    if response_format.get("type") != "json_schema":
        raise ValueError("only json_schema structured output is supported")
    payload = response_format.get("json_schema")
    if not isinstance(payload, dict):
        raise ValueError("json_schema response format is malformed")
    schema = payload.get("schema")
    if not isinstance(schema, dict):
        raise ValueError("json_schema response format has no schema")
    return {
        "name": str(payload.get("name") or "structured_response"),
        "strict": bool(payload.get("strict", True)),
        "schema": schema,
    }


def _normalize_structured_schema(
    schema: dict[str, Any],
    *,
    require_all_properties: bool,
    drop_unsupported_constraints: bool = False,
) -> dict[str, Any]:
    """Translate Pydantic JSON Schema into the providers' supported subset.

    The semantic contract uses discriminated unions, which Pydantic emits as
    oneOf + discriminator, and optional/defaulted fields. OpenAI strict
    structured outputs require every object property to be required. Both
    closed providers support anyOf, while provider-side discriminator/default
    metadata is unnecessary because application-side Pydantic validation
    remains authoritative.
    """

    def visit(value):
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value

        out: dict[str, Any] = {}
        for key, child in value.items():
            if key in {"default", "discriminator"}:
                continue
            if drop_unsupported_constraints and key in {
                "minLength",
                "maxLength",
                "pattern",
                "exclusiveMinimum",
                "exclusiveMaximum",
                "multipleOf",
            }:
                continue
            if key == "oneOf":
                out["anyOf"] = visit(child)
                continue
            if key == "const":
                out["enum"] = [visit(child)]
                continue
            out[key] = visit(child)

        properties = out.get("properties")
        if out.get("type") == "object" and isinstance(properties, dict):
            out["additionalProperties"] = False
            if require_all_properties:
                out["required"] = list(properties)
        return out

    normalized = visit(schema)
    if not isinstance(normalized, dict):
        raise ValueError("normalized structured schema must be an object")
    if normalized.get("type") != "object":
        raise ValueError("structured output root schema must be an object")
    return normalized


def _openai_text_format(response_format: dict[str, Any]) -> dict[str, Any]:
    structured = _schema_components(response_format)
    return {
        "type": "json_schema",
        "name": structured["name"],
        "strict": structured["strict"],
        "schema": _normalize_structured_schema(
            structured["schema"],
            require_all_properties=True,
        ),
    }


def _gemini_request(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    seed: int | None,
    response_format: dict[str, Any] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Consume native SSE while preserving the common response/usage contract.

    A buffered Interactions response can hit an upstream connection deadline
    during long reasoning. Streaming does not change the model, sampling,
    schema, or output allowance. SDK retries are disabled: each recorded call
    is one generation, and the experiment handles infrastructure retries.
    """
    check_provider_stop("gemini")
    from google import genai
    from google.genai import types
    import httpx

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required for the gemini provider")
    base = os.environ.get("GEMINI_BASE_URL", _GEMINI_BASE_URL).rstrip("/")
    # The public option historically includes the API version.
    endpoint, api_version = base.rsplit("/", 1)
    if api_version not in {"v1", "v1beta", "v1alpha"}:
        raise ValueError("GEMINI_BASE_URL must end with an explicit API version")
    url = base + "/models/" + config["model"] + ":streamGenerateContent"
    system_parts = [
        str(item.get("content", ""))
        for item in messages
        if item.get("role") == "system"
    ]
    input_parts = [
        str(item.get("content", ""))
        for item in messages
        if item.get("role") != "system"
    ]
    generation: dict[str, Any] = {
        "system_instruction": "\n\n".join(system_parts),
        "max_output_tokens": int(max_tokens),
        "temperature": float(temperature),
        "candidate_count": 1,
    }
    if top_p is not None:
        generation["top_p"] = float(top_p)
    if top_k is not None:
        generation["top_k"] = int(top_k)
    if seed is not None:
        generation["seed"] = int(seed)
    if response_format is not None:
        structured = _schema_components(response_format)
        generation["response_mime_type"] = "application/json"
        generation["response_json_schema"] = _normalize_structured_schema(
                structured["schema"],
                require_all_properties=False,
                drop_unsupported_constraints=True,
            )
    timeout = float(os.environ.get("GEMINI_TIMEOUT_S", "600"))
    _THREAD_USAGE.response_metadata = {}
    _THREAD_USAGE.request_metadata = {
        "url": url,
        "parameters": {"model": config["model"], "api": "streamGenerateContent",
                       "stream": True, "sdk_generation_attempts": 1,
                       "generation_config": {key: value for key, value in generation.items()
                                             if key not in {"system_instruction", "response_json_schema"}}},
    }
    parts, native_usage = [], {}
    finish_reason = None
    response_id = model_version = None
    chunks = 0
    try:
        with _wall_clock_timeout(timeout):
            with genai.Client(api_key=api_key, http_options=types.HttpOptions(
                base_url=endpoint, api_version=api_version, timeout=int(timeout * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            )) as client:
                for chunk in client.models.generate_content_stream(
                    model=config["model"], contents="\n\n".join(input_parts),
                    config=types.GenerateContentConfig(**generation),
                ):
                    chunks += 1
                    response_id = chunk.response_id or response_id
                    model_version = chunk.model_version or model_version
                    for candidate in chunk.candidates or []:
                        if candidate.content:
                            for part in candidate.content.parts or []:
                                if part.text is not None and not part.thought:
                                    parts.append(part.text)
                        if candidate.finish_reason is not None:
                            finish_reason = str(candidate.finish_reason.value)
                    if chunk.usage_metadata is not None:
                        # Streaming usage is cumulative, never sum chunks.
                        native_usage = chunk.usage_metadata.model_dump(mode="json", exclude_none=True)
        if finish_reason is None:
            raise RuntimeError("Gemini stream ended without a terminal finish reason")
    except Exception as exc:
        _THREAD_USAGE.response_metadata = {
            "api": "streamGenerateContent", "response_id": response_id,
            "model": model_version, "stream_chunks": chunks, "stream_complete": False,
            "partial_text": "".join(parts), "native_usage": native_usage,
            "finish_reason": finish_reason,
        }
        timed_out = _is_timeout_error(exc) or isinstance(exc, httpx.TimeoutException)
        _record_transport_attempt("gemini", 1, "timeout" if timed_out else "transport_error",
                                  timed_out=timed_out)
        stop_on_quota("gemini", exc)
        raise
    _record_transport_attempt("gemini", 1, "success", timed_out=False)
    return {
        "id": response_id, "model": model_version or config["model"],
        "status": "incomplete" if finish_reason == "MAX_TOKENS" else "completed",
        "steps": [{"type": "model_output", "content": [{"type": "text", "text": "".join(parts)}]}],
        "usage": {
            "total_input_tokens": int(native_usage.get("prompt_token_count", 0)),
            "total_output_tokens": int(native_usage.get("candidates_token_count", 0)),
            "total_thought_tokens": int(native_usage.get("thoughts_token_count", 0)),
            "total_cached_tokens": int(native_usage.get("cached_content_token_count", 0)),
            "total_tokens": int(native_usage.get("total_token_count", 0)),
        },
        "transport": {"api": "streamGenerateContent", "response_id": response_id,
                      "finish_reason": finish_reason, "stream_complete": True,
                      "stream_chunks": chunks, "native_usage": native_usage},
    }


def _gemini_response_text(data: dict[str, Any]) -> str:
    status = str(data.get("status", "completed"))
    if status not in {"completed", "COMPLETE", "COMPLETED", "incomplete", "INCOMPLETE"}:
        raise RuntimeError(f"Gemini interaction ended with status {status}")
    parts = [
        str(content["text"])
        for step in data.get("steps", [])
        if step.get("type") == "model_output"
        for content in step.get("content", [])
        if content.get("type") == "text" and content.get("text") is not None
    ]
    if not parts and data.get("output_text") is not None:
        parts = [str(data["output_text"])]
    # Empty or partial text from a returned completion fails contract parsing
    # normally; it must not become an infrastructure retry.
    return "".join(parts)


def _response_text(data: dict[str, Any]) -> str:
    if data.get("status", "completed") not in {"completed", "incomplete"}:
        raise RuntimeError(f"OpenAI response ended with status {data.get('status')}")
    parts = [
        str(content["text"])
        for item in data.get("output", [])
        if item.get("type") == "message"
        for content in item.get("content", [])
        if content.get("type") == "output_text"
    ]
    return "".join(parts)


def _record_usage(
    provider: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    elapsed: float,
    usage: dict[str, int] | None,
) -> None:
    if usage is not None:
        usage["input_tokens"] = usage.get("input_tokens", 0) + input_tokens
        usage["output_tokens"] = usage.get("output_tokens", 0) + output_tokens
        usage["total_tokens"] = usage.get("total_tokens", 0) + total_tokens
    stats = usage_snapshot()
    stats["llm_calls"] += 1
    stats["input_tokens"] += input_tokens
    stats["output_tokens"] += output_tokens
    stats["total_tokens"] += total_tokens
    stats["llm_latency_seconds"] += elapsed
    stats[f"{provider}_calls"] += 1
    stats[f"{provider}_input_tokens"] += input_tokens
    stats[f"{provider}_output_tokens"] += output_tokens
    stats[f"{provider}_total_tokens"] += total_tokens
    stats[f"{provider}_latency_seconds"] += elapsed
    _THREAD_USAGE.stats = stats


def _post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    _THREAD_USAGE.response_metadata = {}
    _THREAD_USAGE.request_metadata = {
        "url": url,
        "parameters": {key: value for key, value in payload.items()
                       if key not in {"messages", "input", "system_instruction", "response_format", "text"}},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if os.environ.get("LLM_HTTP_BACKEND", "urlopen") != "curl":
        request = Request(url, data=body, headers=headers, method="POST")
        with _wall_clock_timeout(timeout):
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))

    cmd = [
        "curl",
        "--silent",
        "--show-error",
        "--fail-with-body",
        "--max-time",
        str(timeout),
        "-X",
        "POST",
        url,
    ]
    for key, value in headers.items():
        cmd.extend(["-H", f"{key}: {value}"])
    cmd.extend(["--data-binary", "@-"])
    result = subprocess.run(
        cmd,
        input=body,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout + 2.0,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace") or result.stdout.decode(
            "utf-8", errors="replace"
        )
        raise TimeoutError(detail.strip() or f"curl exited with {result.returncode}")
    return json.loads(result.stdout.decode("utf-8"))


@contextmanager
def _wall_clock_timeout(seconds: float):
    if (
        seconds <= 0
        or not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    def _raise_timeout(_signum, _frame):
        raise TimeoutError(f"LLM request exceeded {seconds:.1f}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])
