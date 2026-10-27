# Running external baselines

Use the editable checkout installed as in the README for launchers and scoring. Choose one row below and prepare only that method's environment and assets. `reconav/baselines/upstream.json` records source URLs and exact revisions; checkpoints remain external.

| Method | Launcher | Model services |
|---|---|---|
| NavGPT / MapGPT, native or NavGPT +ASG | `reconav.baselines.runners.llm` | Qwen |
| CA-Nav, native | `reconav.baselines.runners.native --method canav` | Qwen |
| GC-VLN, native / +ASG | `reconav.baselines.runners.native --method gcvln` | Qwen; GSAM2 for native |
| SNav, native | `reconav.baselines.runners.native --method snav` | Its own checkpoint; no Qwen |
| DualVLN, native | `reconav.baselines.runners.native --method dualvln` | Its own checkpoint; no Qwen |
| Lang2LTL-2 / LIMP +ASG | `reconav.baselines.runners.formal` | Qwen; T5/embeddings only for Lang2LTL-2 |

Install the core package and set `RECONAV_DATA_ROOT` as in the README. Model/simulator dependencies below apply only to the selected method.

1. Set `RECONAV_EXTERNAL_ROOT`, `RECONAV_MODEL_ROOT` and `RECONAV_HM3D_ROOT` as shown in `.env.example`.
2. Clone each needed upstream into `$RECONAV_EXTERNAL_ROOT/upstream/<directory>`, check out its recorded `commit`, then run `git submodule update --init --recursive`.
   Apply any recorded `patch` from this checkout. For CA-Nav, run `git -C "$RECONAV_EXTERNAL_ROOT/upstream/GroundingDINO-canav" apply "$PWD/reconav/baselines/patches/groundingdino-canav.patch"`.
3. Use a separate environment for each policy and its simulator/model dependencies. Install `requirements.lock` in the core Python 3.11 environment and the method-specific files below in workers. Launchers put the ReCoNav checkout on each worker's `PYTHONPATH`.

For native simulator methods and image-rendering methods (including NavGPT +ASG), prepare both `<scene>.basis.glb` and the original JPEG-textured `<scene>.glb` for the selected scenes in the benchmark manifest. Obtain the validation `habitat` and `glb` components (`hm3d_val_habitat_v0.2`, `hm3d_val_glb_v0.2`) through the [official HM3D downloader](https://github.com/facebookresearch/habitat-sim/blob/main/src_python/habitat_sim/utils/datasets_download.py). Point `RECONAV_HM3D_ROOT` to their `val/` directory.

## Habitat navigation resources

Simulator-based methods also use the optional [Habitat navigation bundle](https://github.com/psiang/ReCoNav/releases/download/v1.0.0/habitat-navigation-v1.0.0.tar.gz). It contains navmeshes, robot settings and exact-start records for Habitat-Sim 0.2.1, 0.2.4 and 0.3.3. Download and extract it once:

```bash
mkdir -p "$RECONAV_EXTERNAL_ROOT/resources" downloads
curl --fail --location \
  https://github.com/psiang/ReCoNav/releases/download/v1.0.0/habitat-navigation-v1.0.0.tar.gz \
  --output downloads/habitat-navigation-v1.0.0.tar.gz
tar -xzf downloads/habitat-navigation-v1.0.0.tar.gz -C "$RECONAV_EXTERNAL_ROOT/resources"
export RECONAV_HABITAT_NAVIGATION_ROOT="$RECONAV_EXTERNAL_ROOT/resources/habitat-navigation"
```

Launchers select the matching version automatically. Set `RECONAV_HABITAT_NAVIGATION_ROOT` to use another location; its default is shown above. See the resource's [data terms](../reconav/baselines/resources/habitat_navigation_DATA_TERMS.md).

To rebuild from the official scenes, use a new cache and the matching Habitat environment. For example, rebuild one scene with Habitat-Sim 0.3.3:

```bash
CUDA_VISIBLE_DEVICES='' RECONAV_CACHE_ROOT="$PWD/.cache-habitat-rebuild" PYTHONPATH="$PWD" \
  /path/to/habitat033/bin/python -m reconav.baselines.runners.prepare_robot \
  --scene 00824-Dd4bFSTQ8gi
```

Omit `--scene` for all scenes. Outputs go under `$RECONAV_CACHE_ROOT/external/robot/<Habitat version>/`. Habitat 0.2.1 also needs `xvfb-run -a` and `LIBGL_ALWAYS_SOFTWARE=1` for its scene loader.

## Policy environments

| Policy | Conda base | Worker configuration |
|---|---|---|
| NavGPT / MapGPT | `environments/habitat033.yml` | `--worker-python /path/to/env/bin/python` |
| CA-Nav | `environments/canav.yml` | `RECONAV_ENV_CANAV=/path/to/env` |
| GC-VLN | `environments/gcvln.yml` | `RECONAV_ENV_GCVLN_RUNTIME_PY39=/path/to/env` |
| DualVLN | `environments/habitat033.yml` | `RECONAV_ENV_DUALVLN_OFFICIAL=/path/to/env` |
| SNav | `environments/habitat033.yml` | `$RECONAV_EXTERNAL_ROOT/runtime_assets/snav/venv` |
| Lang2LTL-2 / LIMP +ASG | `environments/formal.yml` | `--worker-python /path/to/formal-env/bin/python` |

Create each selected method's environment with `conda env create -n reconav-<method> -f <Conda base>`, then activate it before installing its packages below. For SNav, use `--prefix "$RECONAV_EXTERNAL_ROOT/runtime_assets/snav/venv"` instead of `-n`.

On Ubuntu, install the rendering libraries and Xvfb:

```bash
sudo apt-get update
sudo apt-get install build-essential libosmesa6 libgl1-mesa-dri libegl1-mesa libglx-mesa0 xvfb
```

The worker overlays include `worker-geometry.lock` (Shapely 2.0.7, compatible with Python 3.8/3.9). The core solver retains Shapely 2.1.2.

For DualVLN, activate its Python 3.9 / Habitat-Sim 0.3.3 environment and install the CUDA toolkit matching Torch. Install Torch before building FlashAttention:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 setuptools wheel packaging ninja einops==0.8.1
python -m pip install --no-build-isolation \
  -r "$RECONAV_EXTERNAL_ROOT/upstream/InternNav/requirements/habitat_requirements.txt"
python -m pip install --no-build-isolation -r environments/dualvln-overrides.lock
```

Compatibility package pins are under `environments/`. In each NavGPT/MapGPT worker, run `pip install -r environments/navgpt-compat.lock -r environments/rendering.lock`. For CPU image rendering, install system `libosmesa6` and set `PYOPENGL_PLATFORM=osmesa`. CA-Nav and GC-VLN also use Xvfb.

In SNav's environment, run `pip install -r environments/snav-overrides.lock`. In each formal-baseline environment, run `pip install -r environments/formal-compat.lock`; `formal.yml` supplies the Spot temporal-logic library. For Lang2LTL-2, also run `pip install torch==2.5.1 transformers==4.49.0 sentencepiece==0.2.0` and prepare T5 and its embedding resources below.

In GC-VLN's worker, build the pinned PyTorch3D after Torch/CUDA are installed: `pip install --no-deps --no-build-isolation -r environments/gcvln-compat.lock --target "$RECONAV_EXTERNAL_ROOT/compat/gcvln"`.

For CA-Nav, create and activate `environments/canav.yml`, then build its Cython dependencies in this order:

```bash
export SETUPTOOLS_USE_DISTUTILS=stdlib
python -m pip install -r environments/canav-build.lock
python -m pip install --no-deps --no-build-isolation cymem==2.0.11 murmurhash==1.0.15
python -m pip install --no-deps --no-build-isolation preshed==3.0.12 blis==0.7.11
python -m pip install --no-deps Cython==0.29.36
python -m pip install --no-deps --no-build-isolation srsly==2.4.8 thinc==8.2.5
python -m pip install --no-deps --no-build-isolation -r environments/canav.lock
```

For GC-VLN, create and activate `environments/gcvln.yml`, then run:

```bash
python -m pip install -r environments/gcvln-build.lock
python -m pip install --no-deps --no-build-isolation -r environments/gcvln.lock
```

Build each policy's upstream native extensions after installing its overlay.

For GC-VLN on Torch 2.5.1, install the matching CUDA toolkit, then run `python -m reconav.baselines.gcvln.build_glip` in its worker environment. For CA-Nav, build and install GroundingDINO with `pip install --no-deps --no-build-isolation "$RECONAV_EXTERNAL_ROOT/upstream/GroundingDINO-canav"` in the CA-Nav environment.

For GC-VLN's GLIP, run `python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng` before offline episodes.

GC-VLN and CA-Nav also need the `bert-base-uncased` tokenizer and weights in the Hugging Face cache. Set `HF_HOME` to a persistent cache, then prepare the pinned checkpoint and its legacy cache alias before offline runs:

```bash
python - <<'PYTHON'
from pathlib import Path
from huggingface_hub import snapshot_download
p = Path(snapshot_download("bert-base-uncased", revision="86b5e0934494bd15c9632b12f734a8a67f723594",
                          allow_patterns=["config.json", "tokenizer*", "vocab.txt", "model.safetensors", "pytorch_model.bin"]))
refs = p.parents[1] / "refs"
refs.mkdir(exist_ok=True)
(refs / "main").write_text(p.name)
PYTHON
```

Use that same `HF_HOME` for workers and the GSAM2 service; episodes run offline.

DualVLN/SNav use Habitat's Mesa EGL renderer. On Ubuntu, set `LIBGL_ALWAYS_SOFTWARE=1`, `EGL_PLATFORM=surfaceless` and `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json` for the worker; adjust the vendor-file location for your system. Model inference still uses the selected CUDA device.

Place model assets under these external-root locations:

- `runtime_assets/dualvln/InternVLA-N1-DualVLN/`: Git LFS clone of [InternRobotics/InternVLA-N1-DualVLN](https://huggingface.co/InternRobotics/InternVLA-N1-DualVLN), checked out at `a698a9e898b4001621a319e1bc89f02ec715cc86`; run `git lfs pull`. Preserve `.git` and the four full weight shards.
- `runtime_assets/dualvln/checkpoints/depth_anything_v2_metric_hypersim_vits.pth`: download the [Depth Anything V2 Metric Hypersim Small checkpoint](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/blob/3bc65d4e14a6786a61acec16453c50e12bf5f338/depth_anything_v2_metric_hypersim_vits.pth) used by DualVLN's depth encoder.
- GC-VLN: install `third_party/GLIP`, `third_party/ModelServer`, `third_party/Grounded-SAM-2` and their weights following the pinned upstream README. Its upstream tree is the default runtime directory; an optional `runtime_assets/gcvln/` overlay can instead hold the same source layout and weights.
- `artifacts/canav/`: original GroundingDINO/SAM assets under `grounded_sam/`, and the upstream BLIP2/VQA model and processor files under `blip2/` and `vqa/`, from [CA-Nav's checkpoint download](https://drive.google.com/drive/folders/1fHUDDnK-gNNABrcb5u_F93mAQhu8tC8z?usp=sharing). Original R2R reply files are unnecessary for this benchmark.
- `runtime_assets/snav/SNav-7B/` and `runtime_assets/snav/siglip-so400m-patch14-384/`; apply `environments/snav-overrides.lock` on Python 3.9 / Torch 2.5.1-cu124.
- Lang2LTL-2: `$RECONAV_MODEL_ROOT/lang2ltl2/checkpoint-best/`, using the [T5 checkpoint linked by its upstream](https://drive.google.com/drive/folders/1rZl8tblyVj-pZZW4OgbO1NJwMIT2fwx9?usp=sharing).

SNav checkpoints: `TidalYang/SNav-7B@04d236f5125dd8783e0547188b6d664bfe94e45b` and `google/siglip-so400m-patch14-384@9fdffc58afc957d1a03a25b10dba0329ab15c2a3`.

Set an absolute `RECONAV_CACHE_ROOT` (for example, `export RECONAV_CACHE_ROOT="$PWD/.cache"`). For +ASG methods, prepare ASG graphs and starts. Select one task with `--task ID` or Hard44 with `--split hard44`; omit both to prepare all tasks:

```bash
python -m reconav.baselines.runners.prepare --output "$RECONAV_CACHE_ROOT/external/assets" \
  --task 00824_T1_0004
```

Native launchers restore version-matched navmeshes from the downloaded Habitat bundle for the selected scenes automatically. For explicit preparation, add `--native --native-method <method>` to the command above. Only the selected scenes' official meshes are required.

For NavGPT/MapGPT native runs, build viewpoint graphs with the Habitat 0.3.3 worker. Run this from the checkout with its root on `PYTHONPATH`. Choose the same task selection as the run (`--task ID`, `--split hard44`, or neither for all tasks):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD" /path/to/habitat033/bin/python \
  -m reconav.baselines.runners.prepare_native --task 00848_T3_0007
```

Methods that use Qwen need its separate server environment and pinned weights. SNav and DualVLN skip this step:

```bash
CUDA_VISIBLE_DEVICES=0 python -m reconav.serve qwen \
  --model-path /path/to/qwen/checkpoint --external-vision --port 8009
```

GC-VLN native additionally needs a separate **Python 3.10** GSAM2 environment. Create it below, install the CUDA toolkit matching Torch 2.5.1, then run the installation commands from the ReCoNav checkout:

```bash
conda create -n reconav-gcvln-server python=3.10 pip
conda activate reconav-gcvln-server
python -m pip install setuptools wheel -r environments/gcvln-server.lock
python -m pip install --no-deps --no-build-isolation -e "$RECONAV_EXTERNAL_ROOT/upstream/GC-VLN/third_party/ModelServer"
SAM2_BUILD_ALLOW_ERRORS=0 python -m pip install --no-deps --no-build-isolation \
  -e "$RECONAV_EXTERNAL_ROOT/upstream/GC-VLN/third_party/Grounded-SAM-2"
python -m pip install --use-pep517 --no-deps --no-build-isolation \
  -e "$RECONAV_EXTERNAL_ROOT/upstream/GC-VLN/third_party/Grounded-SAM-2/grounding_dino"
```

From the upstream GC-VLN directory, put the ReCoNav checkout on `PYTHONPATH` and run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 \
  python -m reconav.baselines.gcvln.serve --port 7012
```

Inspect the matching launcher and run `preflight`, then `smoke` with explicit task IDs before a full `run`. Preflight checks inputs; smoke executes the policy:

```bash
python -m reconav.baselines.runners.llm --help       # NavGPT / MapGPT, native or +ASG
python -m reconav.baselines.runners.native --help    # CA-Nav / GC-VLN / SNav / DualVLN
python -m reconav.baselines.runners.formal --help    # Lang2LTL-2 / LIMP +ASG

python -m reconav.baselines.runners.llm smoke --method navgpt --condition native \
  --provider local-qwen --base-url http://127.0.0.1:8009/v1 \
  --worker-python /path/to/habitat033/bin/python \
  --task 00848_T3_0007 --output outputs/navgpt-smoke

python -m reconav.baselines.runners.native smoke --method dualvln \
  --task 00824_T1_0001 --gpu 1 --output outputs/dualvln-smoke

python -m reconav.baselines.runners.formal smoke --method limp \
  --base-url http://127.0.0.1:8009/v1 --worker-python /path/to/formal-env/bin/python \
  --task 00824_T1_0004 --output outputs/limp-smoke
```

Use a separate `--output` directory for each run. Set endpoints, credentials, assets and worker environments using the launcher's flags. Both `smoke` and `run` require the selected policy's environment, assets and services.

## Lang2LTL-2 embeddings

Lang2LTL-2 uses Qwen for expression extraction, local T5 for LTL translation, and `text-embedding-3-large` (3,072 dimensions) for REG/SPG grounding. An optional [embedding cache](https://github.com/psiang/ReCoNav/releases/download/v1.0.0/lang2ltl2-embeddings-v1.0.0.zip) contains 5,280 original vectors. Download it separately when preparing this method:

```bash
export RECONAV_LANG2LTL_EMBEDDINGS="$RECONAV_EXTERNAL_ROOT/resources/lang2ltl2/embeddings.zip"
mkdir -p "$(dirname "$RECONAV_LANG2LTL_EMBEDDINGS")"
curl --fail --location \
  https://github.com/psiang/ReCoNav/releases/download/v1.0.0/lang2ltl2-embeddings-v1.0.0.zip \
  --output "$RECONAV_LANG2LTL_EMBEDDINGS"
```

The cache is reused automatically. Set `RECONAV_LANG2LTL_EMBEDDINGS` to store it elsewhere; the default is `$RECONAV_EXTERNAL_ROOT/resources/lang2ltl2/embeddings.zip`. Its contents and [data terms](../reconav/baselines/resources/lang2ltl2_embeddings_DATA_TERMS.md) are separate from the benchmark. New Qwen outputs can introduce uncached expressions; only those require the embedding API. LIMP requires only the Qwen service.

After preparing the Lang2LTL-2 environment and checkpoint, run a real sample:

```bash
export RECONAV_EMBEDDING_API_KEY=YOUR_API_KEY
python -m reconav.baselines.runners.formal smoke --method lang2ltl2 \
  --base-url http://127.0.0.1:8009/v1 --worker-python /path/to/formal-env/bin/python \
  --embedding-budget-usd 0.10 --task 00824_T1_0001 \
  --time-limit-seconds 0 --output outputs/lang2ltl2-one
```

`--embedding-key-file /path/to/key` is an alternative to the environment variable. `OPENAI_API_KEY` is also accepted. The budget is a cumulative USD admission cap using the client's configured rate; it is shared by runs using the same ledger under `$RECONAV_CACHE_ROOT/external/formal/embedding_budget.json`. The default is 0.05. Calls make one attempt and newly obtained vectors persist in the cache. Inspect the ledger and `embedding_calls.json` for usage. If the cap is reached, raise it and repeat the same command with `--retry-failed`.

For cached vectors only, use `--embedding-budget-usd 0`; missing vectors then stop that episode without sending an API request. To move additional vectors between machines, copy the directory selected by `--embedding-cache`. `--grounding-cache` selects REG/SPG's working cache. For a fresh reproduction, omit `--frontend-cache` so that Qwen generates new predictions.

## Full runs and task selection

Use the following launcher arguments for the paper's ten combinations. Prepare the corresponding dependencies above, then run only the rows you need:

| Method / interface | Launcher | Method arguments |
|---|---|---|
| NavGPT native | `reconav.baselines.runners.llm` | `--method navgpt --condition native` |
| MapGPT native | `reconav.baselines.runners.llm` | `--method mapgpt --condition native` |
| NavGPT +ASG | `reconav.baselines.runners.llm` | `--method navgpt --condition asg` |
| CA-Nav native | `reconav.baselines.runners.native` | `--method canav --condition native` |
| GC-VLN native | `reconav.baselines.runners.native` | `--method gcvln --condition native` |
| GC-VLN +ASG | `reconav.baselines.runners.native` | `--method gcvln --condition asg` |
| SNav native | `reconav.baselines.runners.native` | `--method snav --condition native` |
| DualVLN native | `reconav.baselines.runners.native` | `--method dualvln --condition native` |
| Lang2LTL-2 +ASG | `reconav.baselines.runners.formal` | `--method lang2ltl2 --condition asg` |
| LIMP +ASG | `reconav.baselines.runners.formal` | `--method limp --condition asg` |

For example, run NavGPT +ASG on all tasks without an added wall-time cap:

```bash
python -m reconav.baselines.runners.prepare --output "$RECONAV_CACHE_ROOT/external/assets"
python -m reconav.baselines.runners.llm run --method navgpt --condition asg \
  --base-url http://127.0.0.1:8009/v1 --worker-python /path/to/habitat033/bin/python \
  --split all --time-limit-seconds 0 --output outputs/external/navgpt-asg
```

With a remote Qwen service, forward its port to localhost using SSH. The `native` launcher uses `--local-base-url http://127.0.0.1:8009/v1` and `--gpu N`; the `formal` launcher uses `--base-url` and `--worker-python` as in the smoke examples. GC-VLN native also takes `--server-port 7012` for GSAM2. Set the corresponding worker environment paths from the setup table. Repeat the same external command to resume; add `--retry-failed` after repairing an execution error.

Use `smoke --task ID` for one episode, `run --split all` for all 192 tasks or `run --split hard44` for the 44-task subset. Repeat `--task` to select several IDs.

For example, after SNav setup:

```bash
# One task; no Qwen service
python -m reconav.baselines.runners.native smoke --method snav --task 00873_T4_0003 \
  --gpu 0 --output outputs/snav-one

# Hard44 with the paper's uncapped protocol
python -m reconav.baselines.runners.native run --method snav --split hard44 \
  --time-protocol "configs/protocols/hard44.json" --partition hard \
  --gpu 0 --output outputs/snav-hard44
```

Choose timing separately from `--split`. For the paper's original external execution protocol, add `--time-protocol "configs/protocols/main.json"`. For Hard44, add `--time-protocol "configs/protocols/hard44.json" --partition hard`. For fresh full runs, use `--time-limit-seconds 0` and collect the new results with `reconav summarize --runs <run-output> --output outputs/report`; this also produces the retrospective 300-second comparison.

GC-VLN appends poses to `trajectory.jsonl` and exports `trajectory.json` at episode end. Its `runtime_summary.json` reports trajectory-recording time under `recording.trajectory.seconds` and diagnostic JSON write time under `recording.instrumentation_write_seconds`. Both remain included in the episode's wall time.
