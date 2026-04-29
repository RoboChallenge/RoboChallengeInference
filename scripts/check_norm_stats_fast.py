"""Sanity-check ``compute_norm_stats_fast.py`` on a configurable subset of episodes.

What this validates (same core as before, with richer options):

  1. **Expansion correctness**: ``_expand_actions_match_horizon`` on per-episode
     parquet (via ``get_data_file_path``) vs LeRobot ``__getitem__`` with
     ``delta_timestamps``, **element-wise** on ``state`` and expanded ``actions``.

  2. **Sampling**: choose episodes by ``--sampling first|random|strided``,
     explicit ``--episodes 0,2,5``, or the first ``--num-episodes`` only.

  3. **Config defaults**: ``--config-name`` sets ``repo_id`` and ``horizon`` from
     :class:`openpi.training.config.TrainConfig` (same as the norm-stats scripts).

The pure-numpy "oracle" comparison is intentionally avoided: see original docstring
on :class:`openpi.shared.normalize.RunningStats`.

Usage:
    uv run scripts/check_norm_stats_fast.py

    # Random 8 episodes, fixed seed, horizon from model config
    uv run scripts/check_norm_stats_fast.py --config-name pi05_g2_finetune \\
        --num-episodes 8 --sampling random --seed 0

    # Explicit list (sorted internally for a deterministic pass order)
    uv run scripts/check_norm_stats_fast.py --episodes 0,1,2 --horizon 50
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("G2_LEROBOT_REPO_ID", "local/icra_g2_dataset")
# Run from project root: ``uv run scripts/check_norm_stats_fast.py``
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pathlib  # noqa: E402
from typing import Literal  # noqa: E402

import compute_norm_stats_fast as fast  # noqa: E402  # local module
import numpy as np  # noqa: E402
import tyro  # noqa: E402

import openpi.shared.normalize as normalize  # noqa: E402
import openpi.training.config as _config  # noqa: E402


def _to_f32(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=np.float32).reshape(-1)


def _select_episode_indices(
    total_episodes: int,
    *,
    num_episodes: int,
    seed: int,
    sampling: Literal["first", "random", "strided"],
) -> list[int]:
    if total_episodes <= 0:
        raise ValueError("total_episodes must be > 0")
    k = min(num_episodes, total_episodes)
    if k <= 0:
        raise ValueError("num_episodes must be > 0")
    if sampling == "first":
        return list(range(k))
    if sampling == "strided":
        if k == 1:
            return [0]
        # Evenly spread indices in [0, total_episodes - 1], then unique+sort (can shrink if total is tiny)
        idx = [round((total_episodes - 1) * i / (k - 1)) for i in range(k)]
        return sorted(set(idx))
    # random: without replacement, fixed seed
    rng = np.random.default_rng(seed)
    picked = rng.choice(total_episodes, size=k, replace=False)
    return sorted(int(x) for x in picked.tolist())


def _parse_episodes_arg(s: str | None) -> list[int] | None:
    if not s or not s.strip():
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _running_stats(arr: np.ndarray, chunk: int = 50_000) -> normalize.NormStats:
    """Run :class:`RunningStats` over ``arr`` in chunks (same as the fast script)."""
    rs = normalize.RunningStats()
    n = len(arr)
    if n == 0:
        raise ValueError("empty array")
    for i in range(0, n, chunk):
        rs.update(arr[i : i + chunk])
    return rs.get_statistics()


def _diff_report(name: str, a_ns: normalize.NormStats, b_ns: normalize.NormStats) -> None:
    print(f"\n[{name}]")
    for field in ("mean", "std", "q01", "q99"):
        a = np.asarray(getattr(a_ns, field))
        b = np.asarray(getattr(b_ns, field))
        abs_diff = np.abs(a - b)
        argmax = int(np.argmax(abs_diff))
        print(
            f"  {field:4s}  max_abs={abs_diff.max():.6e}  "
            f"@dim={argmax}  a={a[argmax]:+.6f}  b={b[argmax]:+.6f}"
        )


def _materialize_lerobot(
    repo_id: str,
    root: str | pathlib.Path,
    episode_indices: list[int],
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Ground truth ``state`` (N, D) and expanded ``actions`` (N*H, D) from LeRobot.

    Uses a **full** :class:`LeRobotDataset` so global ``episode_data_index`` and
    ``_get_query_indices`` stay consistent for arbitrary (non-consecutive) episode ids.
    """
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    root_s = str(root)
    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root_s)
    fps = meta.fps
    delta = {"actions": [t / fps for t in range(horizon)]}
    ds = lerobot_dataset.LeRobotDataset(
        repo_id,
        root=root_s,
        download_videos=False,
        delta_timestamps=delta,
    )

    ep_from = ds.episode_data_index["from"]
    ep_to = ds.episode_data_index["to"]

    state_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []

    for ep in episode_indices:
        for idx in range(int(ep_from[ep].item()), int(ep_to[ep].item())):
            item = ds[idx]
            state_rows.append(_to_f32(item["state"]))
            action_rows.append(np.asarray(item["actions"], dtype=np.float32))

    s_arr = np.stack(state_rows, axis=0)
    a_arr = np.stack(action_rows, axis=0).reshape(-1, action_rows[0].shape[-1])
    return s_arr, a_arr


def _materialize_fast(
    repo_id: str,
    root: str | pathlib.Path,
    episode_indices: list[int],
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read per-episode parquet (``get_data_file_path``) + ``_expand_actions_match_horizon``."""
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    root = pathlib.Path(root)
    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=str(root))
    state_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []

    for ep in episode_indices:
        p = root / meta.get_data_file_path(ep)
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing episode parquet for episode_index={ep} (expected {p}). "
                "Is the full dataset on disk?"
            )
        cols = fast._read_columns(p)
        state_parts.append(cols["state"])
        action_parts.append(fast._expand_actions_match_horizon(cols["actions"], horizon))

    return np.concatenate(state_parts, axis=0), np.concatenate(action_parts, axis=0)


def main(
    config_name: str = "pi05_g2_finetune",
    *,
    num_episodes: int = 5,
    seed: int = 0,
    sampling: Literal["first", "random", "strided"] = "first",
    episodes: str | None = None,
    horizon: int | None = None,
    atol: float = 1e-5,
) -> None:
    """Compare fast parquet path vs LeRobot on a **subset** of episodes.

    ``G2_LEROBOT_REPO_ID`` is read the same way as the norm-stats scripts (set it before
    launch if the dataset is not the config default). ``--config-name`` provides
    ``repo_id`` and the default action horizon.

    Args:
        config_name: :class:`openpi.training.config.TrainConfig` name.
        num_episodes: How many episodes to take when not using ``--episodes``.
        seed: RNG seed for ``--sampling random``.
        sampling: How to pick episodes when not using ``--episodes``:
            ``first`` | ``random`` (without replacement) | ``strided`` (evenly over all eps).
        episodes: Comma-separated global episode indices (e.g. ``0,2,7``), overrides
            ``num_episodes`` / ``sampling``.
        horizon: Action horizon. Defaults to ``config.model.action_horizon``.
        atol: Reject if ``max|fast-truth|`` on state or actions exceeds this.
    """
    cfg = _config.get_config(config_name)
    if cfg.data.repo_id is None:
        raise ValueError("config.data.repo_id is None")
    repo_id = cfg.data.repo_id
    h = int(cfg.model.action_horizon if horizon is None else horizon)

    root = fast._resolve_dataset_root(repo_id)
    print(f"[check] repo_id={repo_id!r}  root={root}  horizon={h}")

    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=str(root))
    total = int(meta.total_episodes)

    explicit = _parse_episodes_arg(episodes)
    if explicit is not None:
        for e in explicit:
            if e < 0 or e >= total:
                raise ValueError(f"episode {e} out of range [0, {total})")
        ep_list = sorted(explicit)
    else:
        ep_list = _select_episode_indices(total, num_episodes=num_episodes, seed=seed, sampling=sampling)

    print(
        f"[check] sample: {len(ep_list)} episode(s) -> {ep_list if len(ep_list) <= 20 else ep_list[:20] + ['...']}"
    )

    # ---- Fast (parquet) vs LeRobot (ground truth) ----
    f_state, f_act = _materialize_fast(repo_id, root, ep_list, h)
    t_state, t_act = _materialize_lerobot(repo_id, str(root), ep_list, h)

    print(f"[check] state shapes  fast {f_state.shape}  truth {t_state.shape}")
    print(f"[check] actions shapes  fast {f_act.shape}  truth {t_act.shape}")
    assert f_state.shape == t_state.shape, "state shape mismatch"
    assert f_act.shape == t_act.shape, "actions shape mismatch"

    ds = np.abs(f_state - t_state)
    da = np.abs(f_act - t_act)
    print(f"[check] state  max|diff| = {ds.max():.3e}  mean|diff| = {ds.mean():.3e}")
    print(f"[check] actions max|diff| = {da.max():.3e}  mean|diff| = {da.mean():.3e}")

    if float(ds.max()) > atol or float(da.max()) > atol:
        if ds.size:
            b = np.unravel_index(int(np.argmax(ds)), ds.shape)
            print(f"        first bad state idx {b}  fast={f_state[b]:.6f}  truth={t_state[b]:.6f}")
        if da.size:
            b = np.unravel_index(int(np.argmax(da)), da.shape)
            print(f"        first bad act   idx {b}  fast={f_act[b]:.6f}  truth={t_act[b]:.6f}")
        raise SystemExit(
            f"[FAIL] max diff exceeds atol={atol} (tighten data path or use float32 consistently)"
        )

    print(f"[OK ] state and actions match LeRobot (atol={atol})")

    # Optional: same RunningStats on both sides (redundant if arrays match, cheap sanity)
    s_fast = _running_stats(f_state)
    s_tru = _running_stats(t_state)
    a_fast = _running_stats(f_act)
    a_tru = _running_stats(t_act)
    _diff_report("state RunningStats (subset)", s_fast, s_tru)
    _diff_report(f"actions RunningStats (subset, H={h})", a_fast, a_tru)
    print("\n[check] done.")


if __name__ == "__main__":
    tyro.cli(main)
