# Constructing tasks from ASGs

Start with the downloaded benchmark and the core environment from the [README](../README.md). The scene inputs are `scenes/hm3dsem/<scene>/asg/{entity,geometry,topology}.json` and the same scene's `navigation/graph.json.gz`.

## Prepare a working copy

Use a separate data directory for your collection:

```bash
export RECONAV_SOURCE_DATA_ROOT="$RECONAV_DATA_ROOT"
export RECONAV_DATA_ROOT="$PWD/data/custom"
mkdir -p "$RECONAV_DATA_ROOT"
cp -a "$RECONAV_SOURCE_DATA_ROOT/benchmark" "$RECONAV_DATA_ROOT/benchmark"
export RECONAV_TASK_FILE="$RECONAV_DATA_ROOT/benchmark/tasks/hm3dsem/00824-Dd4bFSTQ8gi/custom_T1_0001.json"
cp "$RECONAV_SOURCE_DATA_ROOT/benchmark/tasks/hm3dsem/00824-Dd4bFSTQ8gi/00824_T1_0004.json" \
  "$RECONAV_TASK_FILE"
```

The copied task is a format template. Give it a unique `task_id` matching its filename, then edit its instruction and symbolic requirements.

## Define the task

Choose a start position and targets from the ASG. Coordinates use world `x`, floor height, and `z`; `robot_position.y` stores world `z`. GOAL is an object or region. Object approach and encounter use a 1 m threshold.

| Field | Content |
|---|---|
| `task_id`, `scene_id`, `family` | Unique task ID, existing scene ID, and task family |
| `robot_position`, `start_heading` | Exact start and canonical world yaw |
| `instruction` | Natural-language task |
| `symbolic_ir` | Goal, intermediate visits, order and scoped constraints, with evaluator-selected entity IDs |
| `grounded_dsl` | Executable requirements for the selected binding, including the exact START |
| `uniqueness_proof` | Target-resolution class, target role and qualified candidate IDs |
| `selected_route` | Validated route witness, regenerated below |

Use the released families as examples:

| Family | Requirements | Example task |
|---|---|---|
| `T1_goal` | Goal | `00824_T1_0004` |
| `T2_via` | Intermediate visits and order | `00824_T2_0001` |
| `T3_avoid` | Scoped avoidance | `00824_T3_0003` |
| `T4_count` | Distinct-entity counts | `00824_T4_0001` |
| `T5_prefer` | Preferences | `00824_T5_0009` |
| `T6_composite` | Combined requirements | `00824_T6_0008` |

Keep `symbolic_ir` and `grounded_dsl` consistent. Specify each constraint's scope boundaries and preserve quantifiers, order and preference status when writing the instruction. Instructions refer to entities through descriptions; entity IDs belong in the evaluator annotations.

For target-resolution annotations, enumerate candidates after semantic and spatial matching. Compare the complete requirements for each candidate: `hard_constraint` identifies a target resolved by hard feasibility, and `soft_preference` by preference cost among hard-feasible bindings. Use `qualifier` or `direct` for locally resolved references. Preference violations have unit cost; route length breaks ties within a binding, not between target bindings. Retain the candidate checks with your construction records.

## Generate and validate a route witness

After editing the task, run this from the checkout. It computes the canonical start heading, solves the grounded DSL, replays the resulting route against both DSL and task annotations, and saves the witness:

```bash
python -m reconav.benchmark.construction --task "$RECONAV_TASK_FILE"
```

Use `--task file1.json file2.json` for a batch. Set `--solver-seconds` to change the per-task search budget (default 90 seconds). A successful task prints `"status": "passed"`.

Inspect each instruction against its symbolic requirements and candidate checks. For constraints intended to affect behavior, check a satisfying route and a route that violates the relevant clause while keeping the other requirements satisfied.

## Select and run tasks

Write the collection's task IDs to `benchmark/splits/all.json` and update its manifest counts. Keep the original Hard44 IDs when using the paper's Hard44 split; select your own subsets with `--task-ids`.

Using the model setup from the [README](../README.md#run-reconav), inspect the new task selection:

```bash
reconav run --config configs/qwen.json --methods Qwen-ReCoNav \
  --task-ids custom_T1_0001 --dry-run
```

Remove `--dry-run` and supply the endpoint, tokenizer and a new output directory to execute it. For external baselines, use the corresponding `--task custom_T1_0001` selection and preparation commands in the [baseline guide](baselines.md).

Package the collection with its tasks, scene ASGs, physical navigation graphs, vocabularies, splits, manifest and data terms. Keep construction diagnostics separately from the released task JSONs.
