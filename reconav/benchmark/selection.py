"""Task selection shared by public launchers."""
from __future__ import annotations

import json
from pathlib import Path


def task_ids(tasks_root: Path, split: str = "all", requested=()) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(requested))
    if split == "all":
        return requested
    if split != "hard44":
        raise ValueError(f"Unknown benchmark split: {split}")
    path = Path(tasks_root).resolve().parents[1] / "splits/hard44.json"
    members = json.loads(path.read_text())
    if len(members) != 44 or len(set(members)) != 44:
        raise ValueError(f"Invalid Hard44 split: {path}")
    outside = set(requested) - set(members)
    if outside:
        raise ValueError(f"Tasks outside Hard44: {sorted(outside)}")
    return requested or tuple(members)
