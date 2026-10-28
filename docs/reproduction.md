# Reproducing the paper

Install the core package and download the benchmark as in the [README](../README.md). Run commands from the checkout with the core environment active unless a server environment is specified.

## Open-model checkpoints and services

Use a separate Python 3.12 environment for vLLM:

```bash
python3.12 -m venv .venv-server
.venv-server/bin/python -m pip install -r environments/server.lock
python -m pip install '.[models]'
export RECONAV_MODEL_ROOT=/absolute/path/to/models
```

Download the checkpoint for the selected backbone. Gemma requires accepting its model terms on Hugging Face and logging in with `hf auth login`.

```bash
hf download Qwen/Qwen3.8-27B-FP8 \
  --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
  --local-dir "$RECONAV_MODEL_ROOT/qwen"
hf download google/gemma-4-31B-it \
  --revision 842da3794eaa0b77d5f08bae87a17459d91ff475 \
  --local-dir "$RECONAV_MODEL_ROOT/gemma"
```

Launch one service in a separate terminal:

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv-server/bin/python -m reconav.serve qwen \
  --model-path "$RECONAV_MODEL_ROOT/qwen" --port 8008
# Or:
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv-server/bin/python -m reconav.serve gemma \
  --model-path "$RECONAV_MODEL_ROOT/gemma" --port 8008
```

The profiles use context lengths 192,000 (Qwen) and 131,072 (Gemma), and default tensor parallel sizes 2 and 4. They were exercised on GPUs with about 48 GB memory each. Set `--tensor-parallel-size` and `CUDA_VISIBLE_DEVICES` for your hardware. Inspect the launch command with `--dry-run`. Stop the service before reusing its GPUs for another model.

## ReCoNav, naive baselines and ablations

First run one real task:

```bash
reconav run --config configs/qwen.json --methods Qwen-ReCoNav \
  --task-ids 00824_T3_0003 --qwen-base-url http://127.0.0.1:8008/v1 \
  --tokenizer-path "$RECONAV_MODEL_ROOT/qwen" --output outputs/qwen-one
```

For all six variants and all 192 tasks:

```bash
reconav run --config configs/qwen.json \
  --methods Qwen-Direct Qwen-Direct-Feedback Qwen-Direct-ID-Sequence \
    Qwen-No-Verify-Repair Qwen-Forced-Early Qwen-ReCoNav \
  --split all --qwen-base-url http://127.0.0.1:8008/v1 \
  --tokenizer-path "$RECONAV_MODEL_ROOT/qwen" --output outputs/qwen-all
```

For Gemma, use `configs/gemma.json`, replace the `Qwen-` selectors with `Gemma-`, set the Gemma tokenizer path, and choose `outputs/gemma-all`. Use a single selector to run just one method or ablation.

Use `--split hard44` for Hard44 or `--task-ids ID [ID ...]` for specific tasks. Add `--dry-run` to inspect selection before making model calls. To resume, repeat the identical command with `--resume`. A new configuration or task selection uses a new output directory.

## Closed-model backbones

Install `.[models]`, export the corresponding API key, and select any subset of the six methods. These commands run all six variants; they make paid generation calls.

```bash
export OPENAI_API_KEY=YOUR_API_KEY
reconav run --config configs/gpt.json \
  --methods GPT-Direct GPT-Direct-Feedback GPT-Direct-ID-Sequence \
    GPT-No-Verify-Repair GPT-Forced-Early GPT-ReCoNav \
  --split all --output outputs/gpt-all

export GEMINI_API_KEY=YOUR_API_KEY
reconav run --config configs/gemini.json \
  --methods Gemini-Direct Gemini-Direct-Feedback Gemini-Direct-ID-Sequence \
    Gemini-No-Verify-Repair Gemini-Forced-Early Gemini-ReCoNav \
  --split all --output outputs/gemini-all
```

The profiles specify GPT-5.6 Sol with high reasoning effort and Gemini 3.1 Pro Preview. Use `--task-ids ID` for a single-task check and `--resume` for an interrupted run.

## External baselines

Follow [baseline setup and commands](baselines.md) for the selected policy. The paper includes ten method/interface combinations: NavGPT native/+ASG, MapGPT native, CA-Nav native, GC-VLN native/+ASG, SNav native, DualVLN native, Lang2LTL-2+ASG and LIMP+ASG. External generative calls use Qwen; SNav and DualVLN use their own checkpoints. Lang2LTL-2 also uses local T5 and the embedding cache/API described in the baseline manual.

For a fresh full reproduction, run each selected external method with `--split all --time-limit-seconds 0`. This collects uncapped results for all supported tasks; the report applies the main table's 300-second cutoff afterward and also reports Hard44. Native action limits, solver bounds and model settings remain in effect. Repeat an interrupted external command to resume; after fixing an execution error, add `--retry-failed`.

## Reports

Collect the runs you want to compare, using the same directories passed to their `--output` arguments:

```bash
reconav summarize --runs outputs/qwen-all outputs/gemma-all \
  outputs/external/navgpt-native outputs/external/navgpt-asg \
  --output outputs/report
```

Add other external or closed-model run directories as available. For a one-task check, add `--task-ids ID`; for a Hard44-only run, add `--split hard44`. Omit unrun methods. Each task/method must come from one input run.

The report contains `rows.json`, `summary.json`, `metrics.csv`, `paired.csv` and `report.md`. Resolving/control groups are computed automatically from task metadata. Rates in JSON/CSV are fractions; Markdown shows percentages. Main and Hard44 quality use fixed supported populations (192 ASG / 178 native for all tasks, 44 for Hard44); timeouts, exhausted budgets and terminal runtime failures contribute zero. Ablation quality and costs use completed runs. Prefer uses preference-applicable tasks. Missing or unfinished tasks keep a view pending; finish them and repeat the report command. Per-task errors are retained in `rows.json`. Each experiment also saves its own manifest, predictions, scores and summary.

See the paper for reported results.
