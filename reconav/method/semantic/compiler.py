"""Compile instructions into scene-independent semantic contracts."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from reconav.llm.shared import LLMClient
from reconav.method.semantic.contracts import SceneIndependentSemanticContract

SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "prompts"
    / "scene_independent_semantics"
    / "system.txt"
)


@dataclass(frozen=True)
class SemanticCompilationResult:
    raw_response: str
    candidate: Any | None
    parsed_output: SceneIndependentSemanticContract | None
    parse_error: str | None = None
    timing: dict[str, float] | None = None
    infrastructure_error: str | None = None

    @property
    def parse_success(self) -> bool:
        return self.parsed_output is not None


@dataclass(frozen=True)
class SceneIndependentSemanticCompiler:
    client: LLMClient
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: int = 0
    enable_thinking: bool = False

    def compile(
        self,
        instruction: str,
        context: Any | None = None,
    ) -> SemanticCompilationResult:
        del context
        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        stage_started = time.perf_counter()
        model_started = time.perf_counter()
        raw_response = self.client.generate(
            system_prompt=load_system_prompt(),
            user_prompt=f"Instruction:\n{instruction}",
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            presence_penalty=self.presence_penalty,
            repetition_penalty=self.repetition_penalty,
            seed=self.seed,
            enable_thinking=self.enable_thinking,
            response_format=semantic_contract_response_format(),
        )
        model_seconds = time.perf_counter() - model_started
        parse_started = time.perf_counter()
        try:
            candidate = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            return SemanticCompilationResult(
                raw_response,
                None,
                None,
                str(exc),
                _timing(stage_started, model_seconds, parse_started),
            )
        if not isinstance(candidate, dict):
            return SemanticCompilationResult(
                raw_response,
                candidate,
                None,
                "top-level JSON value must be an object",
                _timing(stage_started, model_seconds, parse_started),
            )
        try:
            parsed = SceneIndependentSemanticContract.model_validate(candidate)
        except ValidationError as exc:
            return SemanticCompilationResult(
                raw_response,
                candidate,
                None,
                str(exc),
                _timing(stage_started, model_seconds, parse_started),
            )
        return SemanticCompilationResult(
            raw_response,
            candidate,
            parsed,
            timing=_timing(stage_started, model_seconds, parse_started),
        )


def load_system_prompt() -> str:
    prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    if not prompt.strip():
        raise ValueError(f"empty Stage-1 semantic prompt: {SYSTEM_PROMPT_PATH}")
    return prompt


def semantic_contract_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "scene_independent_semantic_contract",
            "strict": True,
            "schema": SceneIndependentSemanticContract.model_json_schema(),
        },
    }


def _timing(
    stage_started: float, model_seconds: float, parse_started: float
) -> dict[str, float]:
    finished = time.perf_counter()
    return {
        "model_seconds": model_seconds,
        "parse_seconds": finished - parse_started,
        "normalization_seconds": 0.0,
        "total_seconds": finished - stage_started,
    }
