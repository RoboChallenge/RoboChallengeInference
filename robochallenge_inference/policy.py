"""pi0.5 G2 fine-tune policy wrapper for the RoboChallenge inference pipeline.

This module bridges the RoboChallenge ``InterfaceClient`` observation/action format
with the ``openpi`` policy that was trained via ``pi05_g2_finetune_h*`` configs in
this repository. It is consumed by ``demo.py`` (live Arena run) and ``test.py``
(local mock loop).

State / action layout (must match ``examples/g2/convert_g2_data_to_lerobot.py``):

* Action (24D, all absolute):
    - dim 0-6:   left arm 7 joints
    - dim 7:     left effector
    - dim 8-14:  right arm 7 joints
    - dim 15:    right effector
    - dim 16-20: waist 5 joints
    - dim 21-23: base velocity (vx, vy, vw)

* State (26D):
    - dim 0-6:   left arm 7 joints
    - dim 7:     left effector
    - dim 8-14:  right arm 7 joints
    - dim 15:    right effector
    - dim 16-20: waist 5 joints
    - dim 21-23: robot position (x, y, z)
    - dim 24-25: robot orientation (z, w)
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Camera names returned by ``InterfaceClient.get_state``. Must stay in sync with
# the ``G2_CAMERAS`` list in ``demo.py`` / ``test.py``.
G2_CAMERAS = ("kHeadColor", "kHandLeftColor", "kHandRightColor")

_CAMERA_TO_POLICY_KEY = {
    "kHeadColor": "head_color",
    "kHandLeftColor": "hand_left_color",
    "kHandRightColor": "hand_right_color",
}

STATE_DIM = 26
ACTION_DIM = 24


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def _ensure_openpi_on_path(openpi_src: str | None) -> None:
    """Make ``import openpi`` work without a ``uv pip install -e .``.

    If ``openpi_src`` is given, it is prepended to ``sys.path``. Otherwise we
    try the canonical location for this repo: ``<repo_root>/src``.
    """
    if openpi_src:
        if openpi_src not in sys.path:
            sys.path.insert(0, openpi_src)
        return
    # robochallenge_inference/policy.py  →  <repo_root>/src
    candidate = Path(__file__).resolve().parent.parent / "src"
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def _decode_color_camera(entry: dict, target_w: int, target_h: int) -> np.ndarray:
    """Decode a ``camera/<name>`` payload to RGB ``uint8`` ``(H, W, 3)``."""
    raw: bytes = entry["data"]
    enc = entry.get("encoding", "JPEG")

    if enc == "JPEG":
        buf = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode failed for camera frame")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        # Fallback: let PIL decode whatever the server sent.
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        rgb = np.asarray(img, dtype=np.uint8)

    h, w = rgb.shape[:2]
    if (w, h) != (target_w, target_h):
        rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return rgb


def _build_state_vector(input_data: dict) -> np.ndarray:
    """Assemble the 26-D state vector from one ``get_state`` payload.

    Mirrors ``_build_state`` in ``examples/g2/convert_g2_data_to_lerobot.py``:
    we drop the orientation x/y components (near-constant) and keep (z, w).
    The chassis pose comes from ``slam_pose``; if SLAM is unavailable we fall
    back to zeros (still a valid 26-D vector).
    """
    rp = input_data["robot_position"]
    arm = np.asarray(rp["arm_joint_position"], dtype=np.float32).reshape(14)
    grip = np.asarray(input_data["gripper_position"], dtype=np.float32).reshape(2)
    waist = np.asarray(rp["waist_joint_position"], dtype=np.float32).reshape(5)

    slam = input_data.get("slam_pose") or {}
    pos = np.asarray(slam.get("position", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
    ori = np.asarray(slam.get("orientation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32).reshape(4)

    state = np.concatenate([
        arm[:7],        # dim  0-6   left arm
        grip[:1],       # dim  7     left gripper
        arm[7:14],      # dim  8-14  right arm
        grip[1:2],      # dim 15     right gripper
        waist,          # dim 16-20  waist
        pos,            # dim 21-23  position xyz
        ori[2:4],       # dim 24-25  orientation (z, w)
    ])
    assert state.shape == (STATE_DIM,), f"state shape {state.shape} != ({STATE_DIM},)"
    return state.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Action unpacking
# ---------------------------------------------------------------------------

def _actions_to_robo_dicts(
    actions: np.ndarray,
    head_joint_position: list[float],
) -> list[dict]:
    """Unpack ``(T, 24)`` absolute actions into RoboChallenge joint-action dicts.

    The model does not predict the head joints, so we hold them at the latest
    observed position for the entire action chunk.
    """
    if actions.ndim == 1:
        actions = actions[np.newaxis, :]
    if actions.shape[-1] < ACTION_DIM:
        raise ValueError(f"action dim {actions.shape[-1]} < {ACTION_DIM}")

    head = [float(v) for v in head_joint_position]
    out: list[dict] = []
    for t in range(actions.shape[0]):
        a = actions[t]
        out.append({
            "robot_position": {
                "arm_joint_position": np.concatenate([a[0:7], a[8:15]]).astype(np.float64).tolist(),
                "head_joint_position": head,
                "waist_joint_position": a[16:21].astype(np.float64).tolist(),
            },
            "gripper_position": [float(a[7]), float(a[15])],
            "chassis_velocity": [float(a[21]), float(a[22]), float(a[23])],
        })
    return out


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------

class Pi05Policy:
    """openpi pi0.5 G2 fine-tuned policy.

    Loads an Orbax (JAX) or PyTorch checkpoint directory produced by
    ``scripts/train.py pi05_g2_finetune_h*`` (a folder containing ``params/`` +
    ``assets/`` for JAX, or ``model.safetensors`` for PyTorch).

    The instance is meant to be plugged into the ``GPUClient`` defined in
    ``demo.py``: ``GPUClient`` builds a language prompt from the task targets
    and calls ``policy.run_policy(state, prompt)``.
    """

    IMG_SIZE = 224

    def __init__(
        self,
        checkpoint_path: str,
        *,
        openpi_src: str | None = None,
        train_config_name: str = "pi05_g2_finetune_h50",
        device: str | None = None,
        default_prompt: str = "Pick the item from the shelf and place it into the cart.",
    ):
        _ensure_openpi_on_path(openpi_src)
        try:
            from openpi.policies import policy_config
            from openpi.shared import normalize as openpi_normalize
            from openpi.training import config as train_config
        except ImportError as e:
            raise ImportError(
                "Failed to import openpi. Pass --openpi-src pointing to <repo>/src, "
                "or run with openpi installed (uv sync && uv pip install -e .)."
            ) from e

        ckpt = Path(checkpoint_path).expanduser().resolve()
        if not ckpt.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt}")

        kwargs: dict[str, Any] = {}
        if device:
            kwargs["pytorch_device"] = device

        train_cfg = train_config.get_config(train_config_name)
        data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)

        norm_stats = self._try_load_norm_stats(data_config, ckpt, openpi_normalize)

        logger.info(
            "Loading openpi policy  config=%s  checkpoint=%s",
            train_config_name, ckpt,
        )
        self._policy = policy_config.create_trained_policy(
            train_cfg,
            ckpt,
            default_prompt=default_prompt,
            norm_stats=norm_stats,
            **kwargs,
        )

    @staticmethod
    def _try_load_norm_stats(data_config, ckpt: Path, openpi_normalize):
        """Return fallback norm_stats if the expected path is missing.

        ``policy_config.create_trained_policy`` looks for
        ``<ckpt>/assets/<asset_id>/norm_stats.json``. When training was driven
        by a different ``G2_LEROBOT_REPO_ID`` env var than at inference time,
        ``asset_id`` won't match the on-disk folder. To stay robust we look
        under ``<ckpt>/assets/local/<*>/`` for any ``norm_stats.json`` and load
        it directly. Returning ``None`` lets openpi load stats normally.
        """
        if data_config.asset_id is None:
            return None
        expected = ckpt / "assets" / data_config.asset_id / "norm_stats.json"
        if expected.is_file():
            return None

        assets_local = ckpt / "assets" / "local"
        if not assets_local.is_dir():
            return None
        for sub in sorted(assets_local.iterdir()):
            if (sub / "norm_stats.json").is_file():
                logger.warning(
                    "norm_stats not at %s; falling back to %s "
                    "(set G2_LEROBOT_REPO_ID to silence this).",
                    expected, sub,
                )
                return openpi_normalize.load(sub)
        return None

    # -- public API ---------------------------------------------------------

    def run_policy(self, state: dict, prompt: str) -> list[dict]:
        """Decode cameras → build state → infer → unpack into action dicts.

        Args:
            state: The dict returned by ``InterfaceClient.get_state``.
            prompt: Language instruction built by ``GPUClient._process_prompt``.

        Returns:
            ``list[dict]`` ready to be POSTed via ``InterfaceClient.post_actions``.
        """
        rp = state["robot_position"]
        cameras = state.get("camera") or {}
        sz = self.IMG_SIZE

        missing = [c for c in G2_CAMERAS if c not in cameras]
        if missing:
            raise KeyError(
                f"Missing cameras {missing}; "
                f"call get_state(cameras={list(G2_CAMERAS)}, image_width={sz}, image_height={sz})."
            )

        obs: dict[str, Any] = {}
        for cam in G2_CAMERAS:
            obs[f"observation/{_CAMERA_TO_POLICY_KEY[cam]}"] = _decode_color_camera(cameras[cam], sz, sz)
        obs["observation/state"] = _build_state_vector(state)
        obs["prompt"] = prompt

        try:
            result = self._policy.infer(obs)
        except Exception:
            logger.exception("openpi policy.infer failed")
            raise

        actions = np.asarray(result["actions"], dtype=np.float32)
        return _actions_to_robo_dicts(actions[..., :ACTION_DIM], rp["head_joint_position"])
