"""Launch the paper's open-model service in its separate vLLM environment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["qwen", "gemma"])
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--external-vision", action="store_true",
                        help="Qwen profile for external baselines: 64 images, 131072 context, one sequence")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.external_vision and args.model != "qwen":
        parser.error("external baselines use the Qwen checkpoint")
    model = args.model_path.expanduser().resolve()
    if not (model / "config.json").is_file():
        parser.error("--model-path must contain the pinned checkpoint's config.json")
    qwen = args.model == "qwen"
    parallel = args.tensor_parallel_size or (1 if args.external_vision else 2 if qwen else 4)
    if parallel < 1 or not 1 <= args.port <= 65535:
        parser.error("invalid tensor parallel size or port")
    context = 131072 if args.external_vision or not qwen else 192000
    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(model), "--host", "127.0.0.1", "--port", str(args.port),
               "--served-model-name", "qwen3.8-27b" if qwen else "gemma4-31b",
               "--tensor-parallel-size", str(parallel), "--max-model-len", str(context),
               "--max-num-seqs", "1" if args.external_vision else "4",
               "--max-num-batched-tokens", "4096", "--gpu-memory-utilization", "0.90",
               "--structured-outputs-config", json.dumps({"backend": "xgrammar", "disable_any_whitespace": True}),
               "--reasoning-parser", "qwen3" if qwen else "gemma4",
               "--default-chat-template-kwargs", json.dumps({"enable_thinking": False}),
               "--limit-mm-per-prompt", json.dumps({"image": 64 if args.external_vision else 0,
                                                    "video": 0, "audio": 0}),
               "--enforce-eager", "--trust-remote-code"]
    if args.external_vision:
        command += ["--mm-processor-cache-gb", "0"]
    print("VLLM_USE_MODELSCOPE=0 " + shlex.join(command), flush=True)
    if not args.dry_run:
        # These profiles use local Hugging Face snapshots. Ignore a machine's
        # unrelated default model provider, including for local tokenizer loads.
        return subprocess.call(command, env={**os.environ, "VLLM_USE_MODELSCOPE": "0"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
