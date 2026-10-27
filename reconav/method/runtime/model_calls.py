"""Record every learned call with provider-scoped transport accounting."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import time

from reconav.llm.client import (
    current_model_config,
    transport_attempt_snapshot,
    transport_attempts_since,
    usage_snapshot,
    usage_since,
    request_metadata_snapshot,
    response_metadata_snapshot,
)
from reconav.method.runtime.context_budget import load_tokenizer, request_token_audit
from reconav.method.runtime.scene import digest
from reconav.llm import gemini_transport
from reconav.llm.control import check_provider_stop, stop_on_quota
import json
import uuid


class InputBudgetError(ValueError):
    pass


class ContextBudgetError(InputBudgetError):
    pass


class InputCharacterGuardError(InputBudgetError):
    pass


class RecordedClient:
    def __init__(self, client, *, max_input_characters: int,
                 tokenizer_name_or_path: str | None = None,
                 served_max_model_len: int | None = None,
                 context_safety_margin: int = 4096):
        self.client = client
        model_config = current_model_config(
            getattr(client, "provider", None)
        )
        self.provider = model_config["provider"]
        self.endpoint = model_config.get("endpoint")
        self.max_input_characters = max_input_characters
        if (tokenizer_name_or_path is None) != (served_max_model_len is None):
            raise ValueError("tokenizer and served max model length must be configured together")
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.served_max_model_len = served_max_model_len
        self.context_safety_margin = context_safety_margin
        self.tokenizer = None if tokenizer_name_or_path is None else load_tokenizer(tokenizer_name_or_path)
        self.calls: list[dict] = []
        self.stage = "unspecified"

    @contextmanager
    def at(self, stage):
        previous = self.stage
        self.stage = stage
        try:
            yield self
        finally:
            self.stage = previous

    def preflight(self, system_prompt, user_prompt, response_format=None, *,
                  max_tokens: int, enable_thinking: bool = False):
        if self.provider == "gemini" and gemini_transport.needs_transport(response_format):
            system_prompt, user_prompt, prepared = gemini_transport.prepare(
                system_prompt, user_prompt, {"response_format": response_format, "max_tokens": max_tokens}
            )
            response_format = prepared["response_format"]
            max_tokens = prepared["max_tokens"]
        if self.tokenizer is None:
            return {
                "characters": len(system_prompt) + len(user_prompt),
                "fits": len(system_prompt) + len(user_prompt) <= self.max_input_characters,
                "tokenizer": "not_configured",
                "kind": "character_anomaly_guard_only",
            }
        return request_token_audit(
            tokenizer=self.tokenizer,
            tokenizer_name_or_path=str(self.tokenizer_name_or_path),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=response_format,
            requested_output_tokens=max_tokens,
            served_max_model_len=int(self.served_max_model_len),
            safety_margin=self.context_safety_margin,
            enable_thinking=enable_thinking,
        )

    def generate(self, system_prompt, user_prompt, **kwargs):
        canonical_request = None
        if self.provider == "gemini" and gemini_transport.needs_transport(kwargs.get("response_format")):
            canonical_request = {"system_prompt": system_prompt, "user_prompt": user_prompt,
                                 "parameters": deepcopy(kwargs)}
            system_prompt, user_prompt, kwargs = gemini_transport.prepare(system_prompt, user_prompt, kwargs)
        if self.provider == "gemini":
            # Gemini includes thinking in this cap, even for short ID-only
            # answers. Apply one fixed allowance to every Gemini stage.
            kwargs = {**kwargs, "max_tokens": max(int(kwargs["max_tokens"]), 32768)}
        context = self.preflight(
            system_prompt,
            user_prompt,
            kwargs.get("response_format"),
            max_tokens=int(kwargs["max_tokens"]),
            enable_thinking=bool(kwargs.get("enable_thinking", False)),
        )
        record = {"call_id": uuid.uuid4().hex, "stage": self.stage, "provider": self.provider,
                  "endpoint": self.endpoint,
                  "system_prompt": system_prompt, "user_prompt": user_prompt,
                  "parameters": deepcopy(kwargs), "request_fingerprint": digest([system_prompt, user_prompt, kwargs]),
                  "input_characters": len(system_prompt) + len(user_prompt),
                  "context_admission": context, "provider_called": False}
        self.calls.append(record)
        if canonical_request is not None:
            record["semantic_transport"] = {"version": gemini_transport.VERSION,
                                             "canonical_request": canonical_request}
        if record["input_characters"] > self.max_input_characters:
            record["error"] = "input_character_anomaly_guard_exceeded; no scene truncation performed"
            raise InputCharacterGuardError(record["error"])
        if not context["fits"]:
            record["error"] = "CONTEXT_BUDGET_EXCEEDED; no scene truncation performed"
            raise ContextBudgetError(record["error"])
        usage = usage_snapshot()
        attempts = transport_attempt_snapshot(self.provider)
        started = time.perf_counter()
        try:
            check_provider_stop(self.provider)
            record["provider_called"] = True
            value = self.client.generate(system_prompt, user_prompt, **kwargs)
            if canonical_request is not None:
                record["semantic_transport"]["wire_response"] = value
                try:
                    decoded = gemini_transport.decode(json.loads(value))
                    value = json.dumps(decoded, ensure_ascii=False, allow_nan=False)
                except (ValueError, TypeError) as exc:
                    # Model output errors belong to the caller's parse/gate
                    # result, not to infrastructure retries. Keep invalid wire
                    # output so common contract validation fails explicitly.
                    record["semantic_transport"]["decode_error"] = str(exc)
            record["raw_response"] = value
            return value
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            stop_on_quota(self.provider, exc)
            raise
        finally:
            record["seconds"] = time.perf_counter() - started
            record["usage"] = usage_since(usage)
            record["effective_request"] = request_metadata_snapshot() if record["provider_called"] else {}
            record["provider_response"] = response_metadata_snapshot() if record["provider_called"] else {}
            record["transport_attempts"] = transport_attempts_since(
                self.provider, attempts
            )
