"""Runtime inputs, model responses and method configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
import math

from pydantic import BaseModel, ConfigDict, Field
from reconav.method.semantic.contracts import SceneIndependentSemanticContract

PROVIDER_PREFIXES = {
    "qwen": "Qwen",
    "openai": "GPT",
    "gemini": "Gemini",
}
BACKBONE_PREFIXES = {**PROVIDER_PREFIXES, "gemma": "Gemma"}
PREFIX_BACKBONES = {prefix: backbone for backbone, prefix in BACKBONE_PREFIXES.items()}
PREFIX_PROVIDERS = {prefix: ("qwen" if backbone == "gemma" else backbone)
                    for prefix, backbone in PREFIX_BACKBONES.items()}

MAIN_VARIANTS = (
    "Direct",
    "Direct-Feedback",
    "ReCoNav",
)
ABLATION_VARIANTS = (
    "Direct-ID-Sequence",
    "Forced-Early",
    "No-Verify-Repair",
)
METHOD_VARIANTS = (*MAIN_VARIANTS, *ABLATION_VARIANTS)
_INTERNAL_VARIANTS = ("Before-Repair",)


def normalize_provider(provider: str) -> str:
    value = str(provider).strip().lower()
    aliases = {"gpt": "openai"}
    value = aliases.get(value, value)
    if value not in PROVIDER_PREFIXES:
        raise ValueError(f"unsupported provider {provider!r}")
    return value


def normalize_variant(variant: str) -> str:
    """Use a path-safe selector for the paper's method names."""
    return "No-Verify-Repair" if variant == "No Verify/Repair" else variant


def paper_method_name(variant: str) -> str:
    return "No Verify/Repair" if variant == "No-Verify-Repair" else variant


def method_name(backbone: str, variant: str) -> str:
    backbone = str(backbone).strip().lower()
    backbone = {"gpt": "openai"}.get(backbone, backbone)
    variant = normalize_variant(variant)
    if backbone not in BACKBONE_PREFIXES:
        raise ValueError(f"unsupported model backbone {backbone!r}")
    if variant not in (*METHOD_VARIANTS, *_INTERNAL_VARIANTS):
        raise ValueError(f"unsupported method variant {variant!r}")
    return f"{BACKBONE_PREFIXES[backbone]}-{variant}"


def method_provider(method: str) -> str:
    prefix, separator, variant = str(method).partition("-")
    if not separator or prefix not in PREFIX_PROVIDERS or normalize_variant(variant) not in (*METHOD_VARIANTS, *_INTERNAL_VARIANTS):
        raise ValueError(f"unsupported method {method!r}")
    return PREFIX_PROVIDERS[prefix]


def method_variant(method: str) -> str:
    method_provider(method)
    return normalize_variant(str(method).split("-", 1)[1])


def method_backbone(method: str) -> str:
    method_provider(method)
    return PREFIX_BACKBONES[str(method).split("-", 1)[0]]


def normalize_method(method: str) -> str:
    return method_name(method_backbone(method), method_variant(method))


def methods_for_provider(provider: str) -> tuple[str, ...]:
    return tuple(method_name(provider, variant) for variant in METHOD_VARIANTS)


MAIN_METHODS = tuple(
    method_name(provider, variant)
    for provider in BACKBONE_PREFIXES
    for variant in MAIN_VARIANTS
)
ABLATION_METHODS = tuple(
    method_name(provider, variant)
    for provider in BACKBONE_PREFIXES
    for variant in (*ABLATION_VARIANTS, "ReCoNav")
)
METHODS = tuple(dict.fromkeys((*MAIN_METHODS, *ABLATION_METHODS)))
RUNTIME_VERSION = "paper-main-v33-paper-method-names"


@dataclass(frozen=True)
class NavigationRequest:
    """The only task information that a predictor accepts."""
    task_id: str
    scene_id: str
    instruction: str
    x: float
    floor_height: float
    z: float
    heading: float | None = None

    def __post_init__(self):
        if not self.instruction.strip() or not self.task_id or not self.scene_id:
            raise ValueError("runtime request needs task_id, scene_id and a nonempty instruction")
        if not all(math.isfinite(v) for v in (self.x, self.floor_height, self.z)):
            raise ValueError("START coordinates must be finite")
        if self.heading is not None and not math.isfinite(self.heading):
            raise ValueError("START heading must be finite")

    @property
    def robot_position(self) -> dict[str, float]:
        return {"x": self.x, "floor_height": self.floor_height, "y": self.z}

    @classmethod
    def from_task(cls, task: dict[str, Any]) -> NavigationRequest:
        # Whitelist only. The runtime never receives task family, proof, IDs,
        # canonical_intent, gold DSL, constraints, diagnostics, or reference route.
        p = task["robot_position"]
        heading = task.get("start_heading", p.get("heading"))
        if isinstance(heading, dict):
            heading = heading.get("yaw")
        return cls(str(task["task_id"]), str(task["scene_id"]), str(task["instruction"]),
                   float(p["x"]), float(p["floor_height"]), float(p["y"]),
                   None if heading is None else float(heading))


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class LearnedOccurrence(StrictModel):
    path: str
    asg_kind: Literal["object", "region", "portal", "level"]
    asg_label: str
    grounded_ids: list[int]


class DirectDecision(StrictModel):
    status: Literal["resolved", "unresolved"]
    reason: str
    semantic_contract: SceneIndependentSemanticContract | None
    bindings: list[LearnedOccurrence]


@dataclass(frozen=True)
class DirectAttemptResult:
    raw_response: str
    parsed_output: DirectDecision | None
    parse_error: str | None = None
    infrastructure_error: str | None = None
    timing: dict[str, float] | None = None


class DirectIDSequenceTarget(StrictModel):
    kind: Literal["object", "region", "portal", "level"]
    id: int


class DirectIDSequenceDecision(StrictModel):
    """Direct route baseline: always commit to a non-empty typed ASG-ID sequence."""

    status: Literal["resolved"]
    targets: list[DirectIDSequenceTarget] = Field(min_length=1, max_length=32)


@dataclass(frozen=True)
class RuntimeConfig:
    max_tokens: int = 8192
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: int = 0
    enable_thinking: bool = False
    max_input_characters: int = 512000
    solver_seconds: float = 90.0
    resolution_seconds: float = 360.0
    max_assignments: int = 128
    repair_rounds: int = 2
    feedback_rounds: int = 2
    scene_geometry: Literal["compact"] = "compact"
    tokenizer_name_or_path: str | None = None
    served_max_model_len: int | None = None
    context_safety_margin: int = 4096

    def __post_init__(self):
        if self.max_tokens < 1 or self.max_input_characters < 1 or self.max_assignments < 1:
            raise ValueError("positive token/input/candidate budgets required")
        if (self.tokenizer_name_or_path is None) != (self.served_max_model_len is None):
            raise ValueError("tokenizer and served max model length must be configured together")
        if self.served_max_model_len is not None and self.served_max_model_len < 1:
            raise ValueError("served max model length must be positive")
        if self.context_safety_margin < 0:
            raise ValueError("context safety margin must not be negative")
        if not 0 <= self.repair_rounds <= 2:
            raise ValueError("repair_rounds must be between zero and two")
        if not 0 <= self.feedback_rounds <= 2:
            raise ValueError("feedback_rounds must be between zero and two")
        if self.solver_seconds <= 0 or self.resolution_seconds <= 0:
            raise ValueError("solver and total resolution budgets must be positive")

    def generation_kwargs(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in (
            "max_tokens", "temperature", "top_p", "top_k", "min_p", "presence_penalty",
            "repetition_penalty", "seed", "enable_thinking")}


def response_format(model: type[BaseModel], name: str, count: int | None = None) -> dict:
    schema = model.model_json_schema()
    if count is not None:
        schema["properties"]["bindings"].update(minItems=count, maxItems=count)
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}
