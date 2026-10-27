"""Exact, offline Qwen chat-template accounting for request admission."""
from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
import json
import threading
from typing import Any

from reconav.llm.prompting import build_messages
from reconav.method.runtime.scene import digest

_TOKENIZERS: dict[str, Any] = {}
_TOKENIZER_LOCK = threading.Lock()


def load_tokenizer(name_or_path: str):
    """Load one tokenizer per process; this never contacts a model endpoint."""
    key = str(Path(name_or_path).expanduser())
    with _TOKENIZER_LOCK:
        if key not in _TOKENIZERS:
            from transformers import AutoTokenizer

            local_only = Path(key).exists()
            compatibility = {}
            config_path = Path(key) / "tokenizer_config.json"
            if config_path.is_file():
                metadata = json.loads(config_path.read_text())
                if isinstance(metadata.get("extra_special_tokens"), list):
                    # Transformers 4.x expects a named mapping; 5.x checkpoints
                    # may store a list. Token IDs still come from tokenizer.json.
                    compatibility["extra_special_tokens"] = {
                        name: value for name, value in metadata.items()
                        if name.endswith("_token") and isinstance(value, str)
                        and value in metadata["extra_special_tokens"]
                    }
            _TOKENIZERS[key] = AutoTokenizer.from_pretrained(
                key,
                trust_remote_code=True,
                local_files_only=local_only,
                **compatibility,
            )
        return _TOKENIZERS[key]


def exact_chat_template_tokens(
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    *,
    enable_thinking: bool,
) -> int:
    """Count exactly what vLLM tokenizes from the two chat messages."""
    token_ids = tokenizer.apply_chat_template(
        build_messages(system_prompt, user_prompt),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if isinstance(token_ids, Mapping):
        token_ids = token_ids["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], (list, tuple)):
        if len(token_ids) != 1:
            raise ValueError("token accounting expects exactly one conversation")
        token_ids = token_ids[0]
    return len(token_ids)


def request_token_audit(
    *,
    tokenizer,
    tokenizer_name_or_path: str,
    system_prompt: str,
    user_prompt: str,
    response_format: dict[str, Any] | None,
    requested_output_tokens: int,
    served_max_model_len: int,
    safety_margin: int,
    enable_thinking: bool,
) -> dict[str, Any]:
    """Return the shared runtime/audit accounting for one final request.

    vLLM uses ``response_format`` as a guided-decoding constraint. It does not
    insert that JSON schema into the model's chat prompt, so its characters and
    fingerprint are recorded but it contributes zero prompt tokens.
    """
    messages = build_messages(system_prompt, user_prompt)
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    prompt_tokens = exact_chat_template_tokens(
        tokenizer,
        system_prompt,
        user_prompt,
        enable_thinking=enable_thinking,
    )
    requested = prompt_tokens + int(requested_output_tokens)
    admitted = requested + int(safety_margin)
    return {
        "characters": len(system_prompt) + len(user_prompt),
        "rendered_chat_characters": len(rendered),
        "exact_chat_template_tokens": prompt_tokens,
        "requested_output_tokens": int(requested_output_tokens),
        "total_requested_tokens": requested,
        "safety_margin_tokens": int(safety_margin),
        "admission_tokens": admitted,
        "max_model_len": int(served_max_model_len),
        "headroom_after_margin": int(served_max_model_len) - admitted,
        "fits": admitted <= int(served_max_model_len),
        "tokenizer": tokenizer_name_or_path,
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
        "response_format_in_prompt": False,
        "response_format_characters": 0 if response_format is None else len(
            json.dumps(
                response_format,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        ),
        "response_format_sha256": None if response_format is None else digest(response_format),
    }
