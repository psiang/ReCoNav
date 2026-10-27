"""Original CA-Nav instruction prompt and strict reply grammar."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from reconav.baselines.canav.export_task import CA_NAV_EPISODE_ID


ALLOWED_CONSTRAINTS = {"location constraint", "object constraint", "direction constraint"}
ALLOWED_LANDMARK_ACTIONS = {"approach", "move away", "approach then move away", "approach and then move away"}
ALLOWED_DIRECTIONS = {"left", "right", "forward", "turn left", "turn right", "go forward"}


def official_prompt(upstream_root: Path) -> tuple[str, str]:
    source_path = upstream_root / "vlnce_baselines/common/instruction_tools.py"
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(source_path))
    for node in module.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "prompt_template" for target in targets):
            continue
        value = node.value
        if isinstance(value, ast.JoinedStr) and all(isinstance(part, ast.Constant) for part in value.values):
            prompt = "".join(str(part.value) for part in value.values)
        elif isinstance(value, ast.Constant) and isinstance(value.value, str):
            prompt = value.value
        else:
            raise ValueError("official prompt_template is no longer a static string")
        return prompt, hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    raise ValueError(f"prompt_template not found in {source_path}")


def parse_json_reply(text: str) -> dict[str, Any]:
    value = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        value = fenced.group(1)
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("CA-Nav reply must be a JSON object")
    validate_reply(parsed)
    return parsed


def validate_reply(reply: dict[str, Any]) -> None:
    required = {"destination", "sub-instructions", "state-constraints", "decisions"}
    if set(reply) != required:
        raise ValueError(f"CA-Nav reply keys must be exactly {sorted(required)}")
    if not isinstance(reply["destination"], str) or not reply["destination"].strip():
        raise ValueError("destination must be a non-empty string")
    sub = reply["sub-instructions"]
    if not isinstance(sub, list) or not sub or not all(isinstance(item, str) and item.strip() for item in sub):
        raise ValueError("sub-instructions must be a non-empty list of strings")
    expected_keys = {str(index) for index in range(len(sub))}
    constraints = reply["state-constraints"]
    decisions = reply["decisions"]
    if not isinstance(constraints, dict) or set(constraints) != expected_keys:
        raise ValueError("state-constraints indices must match sub-instructions")
    if not isinstance(decisions, dict) or set(decisions) != expected_keys:
        raise ValueError("decision indices must match sub-instructions")
    for index in expected_keys:
        values = constraints[index]
        if not isinstance(values, list):
            raise ValueError(f"state-constraints[{index}] must be a list")
        for item in values:
            if not isinstance(item, list) or len(item) < 2 or item[0] not in ALLOWED_CONSTRAINTS or not isinstance(item[1], str):
                raise ValueError(f"invalid constraint at stage {index}: {item!r}")
        decision = decisions[index]
        if not isinstance(decision, dict) or set(decision) != {"landmarks", "directions"}:
            raise ValueError(f"invalid decision object at stage {index}")
        for landmark in decision["landmarks"]:
            if (
                not isinstance(landmark, list)
                or len(landmark) != 2
                or not isinstance(landmark[0], str)
                or landmark[1] not in ALLOWED_LANDMARK_ACTIONS
            ):
                raise ValueError(f"invalid landmark decision at stage {index}: {landmark!r}")
        if not isinstance(decision["directions"], list) or any(
            direction not in ALLOWED_DIRECTIONS for direction in decision["directions"]
        ):
            raise ValueError(f"invalid direction decision at stage {index}")


def write_reply(reply: dict[str, Any], output: Path) -> None:
    validate_reply(reply)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({str(CA_NAV_EPISODE_ID): reply}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
