"""Public installation, demonstration and paper reproduction entry points."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reconav import paths


def main(argv=None):
    parser = argparse.ArgumentParser(description="ReCoNav: constrained vision-and-language navigation")
    parser.add_argument("--data-root", type=Path, help="Downloaded data directory containing benchmark/ (default: RECONAV_DATA_ROOT or ./data)")
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="CPU example: actual route solving and scoring; no model or data download")
    smoke.add_argument("--output", type=Path, default=paths.output_root() / "smoke")
    run_command = commands.add_parser("run", help="Run ReCoNav/Direct/ablations with a configured model", add_help=False)
    run_command.add_argument("--config", type=Path, help="Paper backbone profile, e.g. configs/qwen.json")
    summary = commands.add_parser("summarize", help="Summarize newly generated internal and external experiment runs")
    summary.add_argument("--runs", type=Path, nargs="+", required=True)
    summary.add_argument("--split", choices=("all", "hard44"), default="all")
    summary.add_argument("--task-ids", nargs="+", default=[])
    summary.add_argument("--output", type=Path, default=paths.output_root() / "report")
    score = commands.add_parser("score", help="Score a saved prediction on a released task")
    score.add_argument("--task", required=True)
    score.add_argument("--prediction", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    commands.add_parser("benchmark", help="Report the benchmark populations")
    args, extra = parser.parse_known_args(argv)
    data = paths.configure(args.data_root)
    help_requested = any(flag in extra for flag in ("--help", "-h"))
    if args.command != "smoke" and not help_requested:
        if not (data / "benchmark").is_dir():
            parser.error(f"Benchmark not found at {data / 'benchmark'}. Extract the benchmark download and set RECONAV_DATA_ROOT, or pass --data-root before the command.")
    if args.command == "run":
        from reconav.commands.navigation import main as run
        defaults = ["--tasks-root", str(data / "benchmark/tasks/hm3dsem"), "--asg-root", str(data / "benchmark/scenes/hm3dsem"), "--object-vocabulary", str(data / "benchmark/vocabularies/hm3dsem_category_mappings.tsv"), "--output", str(paths.output_root() / "run")]
        if args.config:
            profile = json.loads(args.config.read_text())
            selected = [method for method in profile["methods"] if method.endswith("-ReCoNav")] or profile["methods"][:1]
            if not selected:
                parser.error("The config must declare at least one method.")
            defaults += ["--provider", profile["provider"], "--methods", *selected]
            parameters = {
                "max_tokens": "max-tokens", "temperature": "temperature",
                "top_p": "top-p", "top_k": "top-k", "min_p": "min-p",
                "presence_penalty": "presence-penalty",
                "repetition_penalty": "repetition-penalty", "seed": "seed",
                "solver_seconds": "solver-seconds", "resolution_seconds": "resolution-seconds",
                "max_assignments": "max-assignments", "repair_rounds": "repair-rounds",
                "feedback_rounds": "feedback-rounds", "scene_geometry": "scene-geometry",
                "max_input_characters": "max-input-characters",
                "tokenizer_name_or_path": "tokenizer-path",
                "served_max_model_len": "served-max-model-len",
                "context_safety_margin": "context-safety-margin",
            }
            settings = profile["config"]
            unknown = set(settings) - (parameters.keys() | {"enable_thinking"})
            if unknown:
                parser.error("Unknown runtime config parameter(s): " + ", ".join(sorted(unknown)))
            for key, flag in parameters.items():
                if key in settings and settings[key] is not None:
                    defaults += ["--" + flag, str(settings[key])]
                elif key in settings and key not in {"tokenizer_name_or_path", "served_max_model_len"}:
                    parser.error(f"Runtime config parameter {key} cannot be null.")
            if "enable_thinking" in settings:
                if not isinstance(settings["enable_thinking"], bool):
                    parser.error("Runtime config parameter enable_thinking must be a boolean.")
                defaults += ["--enable-thinking" if settings["enable_thinking"] else "--no-enable-thinking"]
            for key in ("task_data_source", "asg_data_source", "reference_seconds",
                        "task_order", "task_workers", "method_workers"):
                if key in profile:
                    defaults += ["--" + key.replace("_", "-"), str(profile[key])]
            model_flag = {"qwen": "--qwen-model", "openai": "--openai-model", "gemini": "--gemini-model"}[profile["provider"]]
            defaults += [model_flag, profile["model"]["served_model"]]
            if profile["provider"] == "openai" and profile["model"].get("reasoning_effort") is not None:
                defaults += ["--openai-reasoning-effort", profile["model"]["reasoning_effort"]]
        sys.argv = ["reconav", "run", *defaults, *extra]
        run()
    elif extra:
        parser.error("unrecognized arguments: " + " ".join(extra))
    elif args.command == "smoke":
        from reconav.demo import smoke
        print(json.dumps(smoke(args.output), indent=2))
    elif args.command == "summarize":
        from reconav.benchmark.reporting import summarize
        print(json.dumps(summarize(data, args.runs, args.output, args.split, args.task_ids), indent=2))
    elif args.command == "benchmark":
        print((data / "benchmark/manifest.json").read_text())
    elif args.command == "score":
        from reconav.scene.io import load_asg
        from reconav.benchmark.metrics import evaluate_prediction
        import gzip
        matches = list((data / "benchmark/tasks/hm3dsem").glob(f"*/{args.task}.json"))
        if len(matches) != 1:
            parser.error("unknown or ambiguous task ID: " + args.task)
        task = json.loads(matches[0].read_text())
        asg = load_asg(data / "benchmark/scenes/hm3dsem" / task["scene_id"] / "asg")
        opener = gzip.open if args.prediction.suffix == ".gz" else open
        with opener(args.prediction, "rt", encoding="utf-8") as stream:
            prediction = json.load(stream)
        result = evaluate_prediction(task, prediction, asg)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: result[k] for k in ("full_is", "hard_ns", "preference_satisfaction")}, indent=2))


if __name__ == "__main__":
    main()
