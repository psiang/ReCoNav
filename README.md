# ReCoNav

Code for **ReCoNav: Reliable Constrained Long-Horizon Vision-and-Language Navigation via a Determination Engine**.

- **Scene-grounded benchmark** provides 192 tasks across nine HM3DSem scenes to evaluate scoped hard constraints and preferences and their role in target selection.
- **Instruction contract** couples target selection with whole-route satisfaction under scoped hard constraints and preferences.
- **ReCoNav** derives, verifies, and revises interdependent commitments using scene and route evidence through its Determination Engine.

## Citation

If you use ReCoNav or its benchmark, please cite:

```bibtex
@misc{peng2026reconav,
  title  = {{ReCoNav}: Reliable Constrained Long-Horizon Vision-and-Language Navigation via a Determination Engine},
  author = {Peng, Sixiang and Yu, Zehuan and Liu, Wenyi and Xie, Zhensheng and Zhang, Charles and Jiang, Chenhan},
  year   = {2026},
  url    = {https://github.com/psiang/ReCoNav}
}
```

## Benchmark

The benchmark contains **192 tasks in nine HM3DSem scenes**, with final ASGs, physical navigation geometry, annotations and evaluation splits. **Hard44** is the 44-task constraint-resolving subset. Native simulator methods support 178 starts; ASG methods support all 192. Tasks are stored at `benchmark/tasks/hm3dsem/<scene>/<task>.json`; scene assets are at `benchmark/scenes/hm3dsem/<scene>/`.

Download `reconav-benchmark-v1.0.0.tar.gz` from [Hugging Face](https://huggingface.co/datasets/psiang/ReCoNavBench). Accept the dataset terms on its page, then run:

```bash
python -m pip install huggingface_hub
hf auth login
export RECONAV_BENCHMARK_REPO=psiang/ReCoNavBench
export RECONAV_DATA_ROOT="$PWD/data"
hf download "$RECONAV_BENCHMARK_REPO" reconav-benchmark-v1.0.0.tar.gz \
  --repo-type dataset --revision v1.0.0 --local-dir downloads
mkdir -p "$RECONAV_DATA_ROOT"
tar -xzf downloads/reconav-benchmark-v1.0.0.tar.gz -C "$RECONAV_DATA_ROOT"
```

Choose any data location; the archive extracts a `benchmark/` directory. Other paths are configurable in [.env.example](.env.example); export variables in your shell. Run `reconav benchmark` to inspect the task populations.

ReCoNav and its ablations use the released ASGs and geometry. Native simulator baselines and image-rendering baselines additionally require official HM3D scene files. Prepare only the selected baseline's environments and checkpoints. To create tasks using the released ASGs, follow [benchmark construction](docs/benchmark_construction.md).

## Paper setup

ReCoNav is the full method. The paper includes these comparisons:

| Group | Methods |
|---|---|
| Naive baselines | Direct, Direct-Feedback |
| External baselines | NavGPT, MapGPT, CA-Nav, GC-VLN, SNav, DualVLN |
| External baselines +ASG | NavGPT+ASG, GC-VLN+ASG, Lang2LTL-2+ASG, LIMP+ASG |
| Ablations | Direct-ID-Sequence, Forced-Early, No Verify/Repair |

ReCoNav, naive baselines and ablations use the four backbone profiles in `configs/`: Qwen3.8-27B-FP8, Gemma4-31B, GPT-5.6 Sol and Gemini 3.1 Pro Preview. External generative components use Qwen; trained modules retain their released checkpoints. See [reproduction](docs/reproduction.md) and [baseline setup](docs/baselines.md) for commands.

## Run

### Install and smoke

Use Python 3.11 on Linux:

```bash
git clone https://github.com/psiang/ReCoNav.git
cd ReCoNav
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
reconav smoke
```

Smoke solves and scores a synthetic CPU example; no data, model or GPU is needed. Outputs default to `outputs/`.

### Run ReCoNav

Download the benchmark above and install the model clients:

```bash
python -m pip install '.[models]'
```

Choose a backbone independently of the method and task selection:

| Backbone | Profile | Method prefix | Connection |
|---|---|---|---|
| Qwen | `configs/qwen.json` | `Qwen-` | Local model service and matching tokenizer |
| Gemma | `configs/gemma.json` | `Gemma-` | Local model service and matching tokenizer |
| GPT | `configs/gpt.json` | `GPT-` | `OPENAI_API_KEY` |
| Gemini | `configs/gemini.json` | `Gemini-` | `GEMINI_API_KEY` |

For Qwen/Gemma, follow [checkpoint download and server setup](docs/reproduction.md#open-model-checkpoints-and-services), then run in the core environment. For example, run Qwen-ReCoNav on all tasks:

```bash
reconav run --config configs/qwen.json --methods Qwen-ReCoNav \
  --split all --qwen-base-url http://127.0.0.1:8008/v1 \
  --tokenizer-path /path/to/qwen/checkpoint --output outputs/qwen-reconav
```

For Gemma, use `configs/gemma.json`, `Gemma-ReCoNav` and the Gemma endpoint/tokenizer; choose `outputs/gemma-reconav` as the output. Both local backbones use the `--qwen-base-url` flag.

For closed-model APIs, export the corresponding key and run:

```bash
export OPENAI_API_KEY=YOUR_API_KEY
reconav run --config configs/gpt.json --methods GPT-ReCoNav \
  --split all --output outputs/gpt-reconav

# Or use Gemini:
export GEMINI_API_KEY=YOUR_API_KEY
reconav run --config configs/gemini.json --methods Gemini-ReCoNav \
  --split all --output outputs/gemini-reconav
```

To override a profile's model ID, pass `--qwen-model`, `--openai-model` or `--gemini-model`. API endpoints can be changed with `--openai-base-url` or `--gemini-base-url`. For a different local checkpoint, also use its matching tokenizer and context limit (`--served-max-model-len`). See `reconav run --help` for options; explicit flags override profile settings.

Replace `--split all` in any command above to choose the tasks:

| Scope | Arguments |
|---|---|
| All 192 tasks | `--split all` (default) |
| Hard44 | `--split hard44` |
| One task | `--task-ids 00824_T3_0003` |
| Several tasks | `--task-ids 00824_T3_0003 00824_T1_0004` |

Add `--dry-run` to inspect the selection without model calls. To resume, repeat the same command with `--resume`. Use a separate output directory for each model, method selection and task population.

### Naive baselines and ablations

Reuse the commands above and select one or more methods with `--methods`, for example `Qwen-Direct Qwen-Direct-Feedback Qwen-No-Verify-Repair`. Use the matching model prefix to switch backbones. See [method selectors and full runs](docs/reproduction.md#reconav-naive-baselines-and-ablations).

### External baselines and external baselines +ASG

Follow the [baseline guide](docs/baselines.md) to prepare and run the selected method, with native or +ASG inputs. It includes one-task smoke, all/Hard44/task selection, and [Lang2LTL-2 setup](docs/baselines.md#lang2ltl-2-embeddings).

### Summarize new runs

Pass one or more experiment output directories:

```bash
reconav summarize --runs outputs/qwen-reconav outputs/external/navgpt-asg \
  --output outputs/report
```

This writes per-task records, CSV tables and `report.md` for the selected tasks, the 300-second main comparison, Hard44 and internal ablations. See [reproduction](docs/reproduction.md) for selection and resume commands.

## Expected results

`reconav smoke` prints a summary containing:

```json
{
  "status": "passed",
  "solver_status": "SAT",
  "route_scoring_ok": true,
  "model_calls": 0
}
```

The route, score and summary are saved as `route.json`, `score.json` and `summary.json` under `outputs/smoke/` (or `--output`).

## Code structure

```text
reconav/
├── method/                 # ReCoNav, naive baselines and ablations
│   ├── semantic/           # Instruction contracts
│   ├── grounding/          # Bind contracts to the ASG
│   ├── lowering/           # Translate grounded contracts into route programs
│   ├── verification/       # Contract and route verification
│   ├── repair/             # Semantic repair
│   ├── runtime/            # Pipeline and experiment execution
│   └── determination.py    # Determination Engine
├── scene/                  # ASGs and physical navigation geometry
├── solver/                 # Route program solver
├── benchmark/              # Task construction, evaluation and reporting
├── baselines/              # External method adapters
│   ├── runners/            # Shared launch commands
│   └── common/             # Shared inputs, clients and execution utilities
├── llm/                    # Model clients
├── prompts/                # Method prompts
└── commands/               # Navigation CLI implementation
configs/                    # Backbone profiles
environments/               # External baseline environments
docs/                       # Reproduction, baselines and benchmark construction
```

## License

Code: [Apache-2.0](LICENSE). Data, upstream methods and models retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md) and each data archive's `DATA_TERMS.md`.
