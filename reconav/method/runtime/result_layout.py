"""Experiment result names and paths, separate from shared checkpoints."""
from __future__ import annotations

from pathlib import Path

from reconav.method.runtime.models import method_backbone, method_variant


def result_method_id(method: str, asg_data_source: str) -> str:
    from reconav.benchmark.io import validate_run_id

    source = asg_data_source
    backbone = {"openai": "gpt"}.get(method_backbone(method), method_backbone(method))
    value = f"{source}_{backbone}_{method_variant(method)}"
    validate_run_id(value)
    return value


def prediction_path(output, method, scene_id, task_id, *, results_root=None,
                    asg_data_source="hm3dsem") -> Path:
    if results_root is None:
        return Path(output) / "scenes" / scene_id / "predictions" / method / f"{task_id}.json"
    return (Path(results_root) / result_method_id(method, asg_data_source)
            / scene_id / task_id / "result.json")


def prediction_paths(output, method, manifest) -> list[Path]:
    settings = manifest.get("settings", {})
    root = settings.get("results_root")
    if root is None:
        return sorted(Path(output).glob(f"scenes/*/predictions/{method}/*.json"))
    folder = Path(root) / result_method_id(method, settings["asg_data_source"])
    return sorted(folder.glob("*/*/result.json"))
