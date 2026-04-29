# pi0.5 baseline for RoboChallenge / ICRA

This repository is a self‑contained baseline for the **G2 dual‑arm humanoid**
track of [RoboChallenge](https://api.robochallenge.cn/). It packages everything
needed to fine‑tune a [pi0.5](https://www.physicalintelligence.company/blog/pi05)
VLA model on G2 data and to run that model live against the Arena evaluation
platform.

It is a heavily trimmed fork of
[Physical‑Intelligence/openpi](https://github.com/Physical-Intelligence/openpi):
all non‑G2 robots, the FAST autoregressive head, RLDS / DROID data loading,
WebSocket policy serving, and Docker assets have been removed so the codebase
focuses on a single end‑to‑end flow:

```
raw G2 episodes → LeRobot dataset → norm stats → pi0.5 fine‑tune → live G2 inference
```

---

## 1. Repository layout

```
.
├── examples/g2/convert_g2_data_to_lerobot.py   # raw G2 → LeRobot
├── scripts/
│   ├── compute_norm_stats.py                   # reference impl
│   ├── compute_norm_stats_fast.py              # fast parquet impl
│   ├── train.py                                # JAX training entry
│   └── train_pytorch.py                        # PyTorch training entry
├── src/openpi/                                 # trimmed openpi (G2 only)
│   ├── models/, models_pytorch/                # pi0 / pi0.5 implementations
│   ├── policies/g2_policy.py                   # G2 input/output transforms
│   └── training/config.py                      # only G2 + debug configs
├── packages/openpi-client/                     # base policy + image utils
├── robochallenge_inference/                    # Arena client + Pi0.5 wrapper
│   ├── policy.py                               # Pi05Policy bridge
│   ├── demo.py                                 # official Arena entry point
│   ├── test.py                                 # mock‑server smoke test
│   ├── robot/                                  # InterfaceClient + job_loop
│   └── mock_server/                            # local FastAPI mock
├── train.sh                                    # one‑shot train pipeline
└── pyproject.toml
```

Available training configs in `src/openpi/training/config.py`:
| Name | Action horizon | Notes |
|------|----------------|-------|
| `pi05_g2_finetune`     | 50  | recommended default |
| `debug` / `debug_pi05` | —   | tiny configs against `FakeDataConfig` |

---

## 2. Requirements

| Stage             | GPU memory | Tested GPU         |
|-------------------|------------|--------------------|
| Inference         | ≥ 8 GB     | RTX 4090           |
| Fine‑tune (LoRA)  | ≥ 22.5 GB  | RTX 4090           |
| Fine‑tune (full)  | ≥ 70 GB    | A100 80GB / H100   |

* Linux (tested on Ubuntu 22.04), CUDA 12, Python 3.11.
* Dependencies are pinned via `uv`; install it first:
  <https://docs.astral.sh/uv/getting-started/installation/>.

---

## 3. Installation

```bash
git clone <this-repo-url> pi05_g2_baseline
cd pi05_g2_baseline

# Install python deps (jax + cuda12, torch, lerobot, openpi-client, ...)
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Optional: extras needed by `robochallenge_inference/mock_server/mock_server.py`.
GIT_LFS_SKIP_SMUDGE=1 uv sync --group mock
```

`uv sync` creates `.venv/` automatically. **Use the same env for training,
`scripts/train.py`, `scripts/compute_norm_stats[_fast].py`, and the
`robochallenge_inference/` entry points** — `Pi05Policy` imports `openpi.*`
directly and needs the full ML stack (jax / flax / torch / transformers /
lerobot / safetensors / orbax) provided by this `pyproject.toml`.

Prefix any subsequent command with `uv run`, or activate once with
`source .venv/bin/activate`.

---

## 4. End‑to‑end pipeline

The full pipeline is in `train.sh`. Each step is reproduced below with more
context. `G2_LEROBOT_REPO_ID` is the LeRobot dataset id used by the converter,
the norm‑stats script, and the training config — keep it consistent across all
three steps.

```bash
export G2_LEROBOT_REPO_ID="local/icra_g2_dataset"
```

### 4.1 Convert raw G2 data → LeRobot

`examples/g2/convert_g2_data_to_lerobot.py` reads the official G2 dump
(image folders + per‑episode HDF5 + `lang.json`) and writes a LeRobot dataset
at `~/.cache/huggingface/lerobot/$G2_LEROBOT_REPO_ID`.

It produces:

* `actions` — **24‑D absolute** vector
  `[left_arm_7, left_grip, right_arm_7, right_grip, waist_5, base_vxvyvw_3]`
* `observation.state` — **26‑D** vector
  `[left_arm_7, left_grip, right_arm_7, right_grip, waist_5, slam_xyz_3, slam_zw_2]`
* three 224×224 RGB streams: `head_color`, `hand_left_color`, `hand_right_color`
* a target‑centric prompt:
  `"Target: <a>. Pick the <a> from the shelf and place it into the cart."`

```bash
uv run examples/g2/convert_g2_data_to_lerobot.py \
    --data-dir <your_g2_icra_data_dir> \
    --num-workers 32 \
    --queue-size 32
```

### 4.2 Compute normalization statistics

The training pipeline normalizes `state` / `actions` with quantile stats stored
under `assets/<config_name>/<repo_id>/norm_stats.json`.

Two equivalent implementations are provided:

```bash
# Fast: read state/actions directly from LeRobot parquet files (recommended).
uv run scripts/compute_norm_stats_fast.py --config-name pi05_g2_finetune

# Reference: full HuggingFace pipeline with image decoding (slower).
uv run scripts/compute_norm_stats.py --config-name pi05_g2_finetune
```

Both write the same `norm_stats.json` and a sanity report; you can verify
equivalence with `scripts/check_norm_stats_fast.py`.

### 4.3 Fine‑tune pi0.5 on G2 data

```bash
# JAX (default).
uv run scripts/train.py pi05_g2_finetune --exp-name g2_finetune_run_1

# PyTorch (alternative).
uv run scripts/train_pytorch.py pi05_g2_finetune --exp-name g2_finetune_torch
```

The first run downloads the pi0.5 base checkpoint from
`gs://openpi-assets/checkpoints/pi05_base/params` (configured in
`weight_loaders.CheckpointWeightLoader`). Checkpoints land under
`checkpoints/<config_name>/<exp_name>/<step>/`.

To use Weights & Biases logging, set `WANDB_API_KEY`; otherwise edit the
config to set `wandb_enabled=False` (the `debug` config already does this).

For a multi‑GPU sharded run, set `fsdp_devices` in the training config or via
the CLI (`scripts/train.py pi05_g2_finetune --fsdp-devices 4 ...`).

---

## 5. Inference on RoboChallenge G2

The `robochallenge_inference/` folder contains a turnkey Arena client. The
heavy lifting is in `policy.py::Pi05Policy`, which:

1. ensures `import openpi` works from a checkout (`--openpi-src`),
2. loads either an Orbax (`params/`) or PyTorch (`model.safetensors`) checkpoint
   produced by step 4.3,
3. converts a single `InterfaceClient.get_state` payload (cameras + joints +
   SLAM) into the 26‑D state and three 224×224 RGB streams,
4. runs `policy_config.create_trained_policy(...).infer(...)` with the same
   target‑centric prompt that was used during training,
5. unpacks the `(T, 24)` absolute action chunk into the joint‑mode dicts that
   `InterfaceClient.post_actions` expects.

`demo.py` is the official entry point used during evaluation; `test.py` is a
local smoke test that drives the same code paths against a FastAPI mock.

### 5.1 Official baseline checkpoint (Hugging Face)

Organizers provide a **fine‑tuned G2 pi0.5 baseline** (JAX / Orbax only; there
is **no** PyTorch `model.safetensors` in this release) here:

**<https://huggingface.co/RoboChallenge/icra_wbc_baseline>**

**Download** (from the repo root; requires `huggingface_hub`, already a
dependency of this project):

```bash
uv run huggingface-cli download RoboChallenge/icra_wbc_baseline \
    --repo-type model \
    --local-dir ./checkpoints/icra_wbc_baseline
```

Use `huggingface-cli login` or set `HF_TOKEN` if access is restricted.

**`--checkpoint` —** path to the **innermost step directory** that matches a
local training run (§4.3): for this Hub model it must directly contain the JAX
**`params/`** tree (Orbax), and usually **`assets/`** with `norm_stats.json`.
Open the `Files` tab on the Hub if the archive uses extra nesting; pick the
folder at that leaf level. (PyTorch checkpoints use `model.safetensors`
instead; this baseline does not ship that layout.)

**`--train-config`** must be the **same OpenPI config name** the checkpoint
was trained with. For weights produced with this repository’s default G2
recipe, use `pi05_g2_finetune` (see `src/openpi/training/config.py`).

Example after download (adjust `<path_to_step>`). For JAX checkpoints,
`--device` is ignored; omit it or leave it as you prefer:

```bash
uv run python robochallenge_inference/test.py \
    --checkpoint ./checkpoints/icra_wbc_baseline/<path_to_step> \
    --train-config pi05_g2_finetune \
    --openpi-src "$(pwd)/src" \
    --targets Pepsi

uv run python robochallenge_inference/demo.py \
    --user_token <token> \
    --run_id <Run_ID> \
    --checkpoint ./checkpoints/icra_wbc_baseline/<path_to_step> \
    --train-config pi05_g2_finetune \
    --openpi-src "$(pwd)/src" \
    --action-freq 30
```

### 5.2 Local smoke test (mock server)

```bash
# Make sure the mock-server extras are installed (one-time):
GIT_LFS_SKIP_SMUDGE=1 uv sync --group mock

# Terminal 1 — start the mock direct‑robot API on 127.0.0.1:9098.
uv run python robochallenge_inference/mock_server/mock_server.py

# Terminal 2 — run a single mock job through the policy.
uv run python robochallenge_inference/test.py \
    --checkpoint <your_model_checkpoint_path> \
    --train-config pi05_g2_finetune \
    --openpi-src "$(pwd)/src" \
    --device cuda \
    --targets Pepsi
```

The script exits cleanly after the mock finishes one job; logs go to stderr.

### 5.3 Live Arena run

After your evaluation request is accepted on the RoboChallenge web console,
copy the **Run ID** from *My Submissions* and run from the repo root:

```bash
uv run python robochallenge_inference/demo.py \
    --user_token <your_robochallenge_user_token> \
    --run_id <Run_ID_from_My_Submissions> \
    --checkpoint <your_model_checkpoint_path> \
    --train-config pi05_g2_finetune \
    --openpi-src "$(pwd)/src" \
    --device cuda \
    --action-freq 30
```

The process keeps polling Arena for assigned jobs, executes them sequentially,
and exits when the collection is drained. Logs are written to `mylogfile.log`
(see `demo.py` logging config).

> **Prompt consistency.** `GPUClient._process_prompt` rebuilds the exact prompt
> format used by `convert_g2_data_to_lerobot.py`
> (`"Target: <a>[ and <b>]. Pick the <a> [and the <b>] from the shelf and place
> [it|them] into the cart."`). If you change the converter, mirror the change
> in `demo.py` to keep training and inference aligned.

> **Targets.** `robot/job_worker.py::job_loop` resolves the per‑job
> `target_objects` from `robochallenge_inference/robot/config.yaml`
> (`task_id` → `index` → list of product names). Update that file before
> registering a new task on the platform.

The full Arena HTTP API (`/state`, `/actions`, `/status`, `/stop_motion`,
`/goto_navi_position`, …) and the G2 joint limits are documented at
<https://api.robochallenge.cn/> and inline in
`robochallenge_inference/robot/interface_client.py`.

---

## 6. Action / state contract (must stay aligned)

The same layout is enforced by the converter, the training transform
(`src/openpi/policies/g2_policy.py`), and the inference wrapper. Changing it
in one place silently breaks the others.

**Action (24‑D, all absolute):**
```
0–6   left arm joints (rad)
7     left gripper position ∈ [-0.91, 0]
8–14  right arm joints (rad)
15    right gripper position ∈ [-0.91, 0]
16–20 waist joints (rad)
21–23 chassis velocity (vx, vy, wz)
```

**State (26‑D):**
```
0–6   left arm joints
7     left gripper
8–14  right arm joints
15    right gripper
16–20 waist
21–23 SLAM position (x, y, z)
24–25 SLAM orientation (z, w)   # quaternion x/y are dropped
```

The model does **not** predict head joints; `Pi05Policy` holds them at the
last observed value when expanding the action chunk.

---

## 7. Troubleshooting

* **`Failed to import openpi`** — pass `--openpi-src "$(pwd)/src"` to
  `demo.py` / `test.py`, or run `uv pip install -e .` first.
* **Norm stats not found** — the training config uses
  `<assets_base_dir>/<config_name>/<repo_id>/norm_stats.json`; rerun
  `compute_norm_stats[_fast].py` with the same `G2_LEROBOT_REPO_ID` you used
  during conversion.
* **Norm stats path mismatch at inference** — `Pi05Policy._try_load_norm_stats`
  falls back to any `assets/local/<*>/norm_stats.json` inside the checkpoint;
  set `G2_LEROBOT_REPO_ID` to silence the warning.
* **CUDA OOM during fine‑tuning** — reduce `batch_size`, increase
  `fsdp_devices`, or swap to one of the smaller `action_horizon` configs.

---

## 8. License & credits

* Code: Apache 2.0 (see `LICENSE`). Pi0.5 / Pi0 weights and PaliGemma assets
  retain their upstream licenses (`LICENSE_GEMMA.txt`).
* This repo is derived from
  [Physical‑Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
  and uses the [LeRobot](https://github.com/huggingface/lerobot) dataset
  format.
* RoboChallenge platform & G2 robot: <https://api.robochallenge.cn/>.
