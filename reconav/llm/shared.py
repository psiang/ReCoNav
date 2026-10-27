"""Clean adapter over the repository's existing LLM client."""

from __future__ import annotations

from typing import Any, Protocol

from reconav.llm.client import LLM_SEED, chat_completion
from reconav.llm.prompting import build_messages


class LLMClient(Protocol):
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int,
        temperature: float = 0,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        seed: int | None = LLM_SEED,
        enable_thinking: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        ...


class SharedLLMClient:
    """Use one configured provider for every learned call in a method run."""

    def __init__(self, provider: str | None = None) -> None:
        self.provider = provider

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int,
        temperature: float = 0,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        seed: int | None = LLM_SEED,
        enable_thinking: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        return chat_completion(
            messages=build_messages(system_prompt, user_prompt),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            seed=seed,
            enable_thinking=enable_thinking,
            response_format=response_format,
            provider=self.provider,
        )
