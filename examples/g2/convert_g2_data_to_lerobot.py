"""
Convert raw G2 robot data to LeRobot format for pi05 fine-tuning.
Uses multiprocessing to decode/resize video frames in parallel.

Dimension layout aligned with pi05 pretrained action space:

Action (24D):
  dim 0-6:   left arm 7 joints
  dim 7:     left effector
  dim 8-14:  right arm 7 joints
  dim 15:    right effector
  dim 16-20: waist 5 joints
  dim 21-23: base velocity (vx, vy, vw)

State (26D):
  dim 0-6:   left arm 7 joints
  dim 7:     left effector
  dim 8-14:  right arm 7 joints
  dim 15:    right effector
  dim 16-20: waist 5 joints
  dim 21-23: robot position (x, y, z)
  dim 24-25: robot orientation (z, w)

Usage:
  uv run examples/g2/convert_g2_data_to_lerobot.py --data-dir ...
  LeRobot ``repo_id`` comes from env ``G2_LEROBOT_REPO_ID`` (default ``local/g2_dataset_absolute``).
"""

from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
import json
import multiprocessing as mp
from pathlib import Path
import queue
import re
import shutil
import threading

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from tqdm import tqdm
import tyro

import openpi.training.config as _config

ACTION_DIM = 24
STATE_DIM = 26

CAM_NAMES = ("head_color", "hand_left_color", "hand_right_color")
TARGET_SIZE = (224, 224)

_GRASP_RE = re.compile(r"grasp\s+(.+?)\s*$", re.IGNORECASE)
_LEAD_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


def _split_targets(raw: str) -> list[str]:
    """Split a grasp phrase into per-item names, preserving duplicates.

    The raw English phrase comes from the "Grasp ..." sub-step, e.g.
    ``"the Pepsi"`` or ``"the Mirinda and the Mirinda"``. Each ``and``-
    separated chunk is one physical item (the dataset uses bimanual grasps,
    so duplicates like ``Mirinda and Mirinda`` mean two bottles, one per
    hand) and must NOT be deduplicated. Leading articles are stripped so the
    target tokens stay compact.
    """
    parts = re.split(r"\s+and\s+", raw, flags=re.IGNORECASE)
    items: list[str] = []
    for p in parts:
        item = _LEAD_ARTICLE_RE.sub("", p.strip()).rstrip(".").strip()
        if item:
            items.append(item)
    return items


def _build_language_instruction(extra_steps: list[str]) -> str:
    """Build a target-centric prompt aligned with pi05 pretrained style.

    Pulls the target item(s) out of the "Grasp the ..." sub-step and forms a
    single imperative sentence with the target(s) repeated once for emphasis.
    Falls back to comma-joined steps if the grasp line is missing.
    """
    items: list[str] = []
    for s in extra_steps:
        m = _GRASP_RE.match(str(s).strip())
        if m:
            items = _split_targets(m.group(1))
            break

    if not items:
        return ", ".join(str(s) for s in extra_steps)

    if len(items) == 1:
        target_phrase = items[0]
        action_phrase = f"Pick the {items[0]} from the shelf and place it into the cart."
    else:
        target_phrase = " and ".join(items)
        action_object = " and ".join(f"the {it}" for it in items)
        action_phrase = f"Pick {action_object} from the shelf and place them into the cart."

    return f"Target: {target_phrase}. {action_phrase}"


def _decode_video(video_path: str, target_w: int, target_h: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    return np.stack(frames) if frames else np.empty((0, target_h, target_w, 3), dtype=np.uint8)


def _build_state(step) -> np.ndarray:
    """Build 26D state vector: drop orientation x/y (near-constant)."""
    joint = step["state/joint/position"][:]          # 14D: left_arm(7) + right_arm(7)
    left_eff = step["state/left_effector/position"][:]   # 1D
    right_eff = step["state/right_effector/position"][:]  # 1D
    waist = step["state/waist/position"][:]          # 5D
    robot_pos = step["state/robot/position"][:]      # 3D
    robot_ori = step["state/robot/orientation"][:]   # 4D (x, y, z, w)

    return np.concatenate([
        joint[:7],      # dim 0-6:   left arm
        left_eff,       # dim 7:     left effector
        joint[7:14],    # dim 8-14:  right arm
        right_eff,      # dim 15:    right effector
        waist,          # dim 16-20: waist
        robot_pos,      # dim 21-23: robot position
        robot_ori[2:],  # dim 24-25: robot orientation (z, w only)
    ]).astype(np.float32)


def _build_action(step) -> np.ndarray:
    """Build 24D action vector aligned with pi05 pretrained layout."""
    joint = step["action/joint/position"][:]          # 14D
    left_eff = step["action/left_effector/position"][:]   # 1D
    right_eff = step["action/right_effector/position"][:]  # 1D
    waist = step["action/waist/position"][:]          # 5D
    vel = step["action/robot/velocity"][:]            # 6D → pick vx, vy, vw

    return np.concatenate([
        joint[:7],      # dim 0-6:   left arm
        left_eff,       # dim 7:     left effector
        joint[7:14],    # dim 8-14:  right arm
        right_eff,      # dim 15:    right effector
        waist,          # dim 16-20: waist
        np.array([float(vel[0]), float(vel[1]), float(vel[5])]),  # dim 21-23: vx, vy, vw
    ]).astype(np.float32)


def read_all_states_actions(h5_path: str, num_timesteps: int) -> tuple[np.ndarray, np.ndarray]:
    states = np.empty((num_timesteps, STATE_DIM), dtype=np.float32)
    actions = np.empty((num_timesteps, ACTION_DIM), dtype=np.float32)

    with h5py.File(h5_path, "r") as h5f:
        for t in range(num_timesteps):
            ts = str(t)
            if ts not in h5f:
                return states[:t], actions[:t]
            step = h5f[ts]
            states[t] = _build_state(step)
            actions[t] = _build_action(step)

    return states, actions


def preprocess_episode(h5_path: str) -> dict | None:
    episode_dir = Path(h5_path).parent

    meta_path = episode_dir / "meta_info.json"
    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)
    text_data = json.loads(meta.get("text", "{}"))
    extra_steps = text_data.get("extra", [])
    if isinstance(extra_steps, list):
        language_instruction = _build_language_instruction(extra_steps)
    else:
        language_instruction = str(extra_steps) if extra_steps else ""

    video_dir = episode_dir / "observations" / "videos"
    for cam in CAM_NAMES:
        if not (video_dir / f"{cam}.mp4").exists():
            return None

    cam_frames = {}
    for cam in CAM_NAMES:
        arr = _decode_video(str(video_dir / f"{cam}.mp4"), TARGET_SIZE[0], TARGET_SIZE[1])
        if len(arr) == 0:
            return None
        cam_frames[cam] = arr

    min_video_len = min(len(v) for v in cam_frames.values())

    with h5py.File(h5_path, "r") as h5f:
        num_timesteps = len(h5f.keys())
    num_frames = min(min_video_len, num_timesteps)
    states, actions = read_all_states_actions(h5_path, num_frames)
    num_frames = min(num_frames, len(states))

    for cam in CAM_NAMES:
        cam_frames[cam] = cam_frames[cam][:num_frames]

    return {
        "cam_frames": cam_frames,
        "states": states,
        "actions": actions,
        "language_instruction": language_instruction,
        "num_frames": num_frames,
        "episode_dir": str(episode_dir),
    }


def main(
    data_dir: str,
    *,
    push_to_hub: bool = False,
    num_workers: int | None = None,
    queue_size: int | None = None,
):
    """Convert raw G2 episodes into a LeRobot dataset.

    Architecture:
        A producer thread drives a ``ProcessPoolExecutor`` that decodes
        episodes in parallel and pushes finished episodes into a bounded
        queue. The main thread is the consumer and writes those episodes
        into LeRobot. Decoding and writing run truly in parallel; the queue
        provides backpressure so peak RAM stays bounded.

    Args:
        data_dir: Root directory containing raw episodes.
        push_to_hub: Whether to push the resulting dataset to HF Hub.
        num_workers: Decoder worker processes. Defaults to the training
            config; lower this on low-RAM hosts.
        queue_size: Max number of fully-decoded episodes buffered between the
            producer and the consumer. Defaults to ``num_workers``. Peak
            episode-frame RAM is roughly
            ``(num_workers + queue_size) * <single-episode size>``.
    """
    config = _config.get_config("pi05_g2_finetune_h50")
    if num_workers is None:
        num_workers = config.num_workers
    if queue_size is None:
        queue_size = num_workers * 5

    repo_id = _config.g2_lerobot_repo_id()

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)
    data_dir_path = Path(data_dir)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="g2",
        fps=30,
        features={
            "head_color": {
                "dtype": "image",
                "shape": (TARGET_SIZE[1], TARGET_SIZE[0], 3),
                "names": ["height", "width", "channel"],
            },
            "hand_left_color": {
                "dtype": "image",
                "shape": (TARGET_SIZE[1], TARGET_SIZE[0], 3),
                "names": ["height", "width", "channel"],
            },
            "hand_right_color": {
                "dtype": "image",
                "shape": (TARGET_SIZE[1], TARGET_SIZE[0], 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (STATE_DIM,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (ACTION_DIM,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    h5_paths = sorted(data_dir_path.glob("**/aligned_joints.h5"))
    h5_paths = [p for p in h5_paths if "record" not in p.parts]
    max_inflight = num_workers + 2
    print(
        f"Found {len(h5_paths)} episodes, using {num_workers} workers "
        f"(max_inflight={max_inflight}, queue_size={queue_size})"
    )

    # Producer (decoder) -> bounded queue -> consumer (LeRobot writer).
    # Item layout: (ep_path, ep_data | None, exception | None).
    result_queue: queue.Queue = queue.Queue(maxsize=queue_size)
    SENTINEL = object()
    producer_error: list[BaseException] = []

    def producer() -> None:
        try:
            pending_iter = iter(h5_paths)
            with ProcessPoolExecutor(
                max_workers=num_workers, mp_context=mp.get_context("spawn")
            ) as pool:
                inflight: dict = {}
                for _ in range(max_inflight):
                    try:
                        p = next(pending_iter)
                    except StopIteration:
                        break
                    inflight[pool.submit(preprocess_episode, str(p))] = p

                while inflight:
                    done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                    for fut in done:
                        ep_path = inflight.pop(fut)
                        ep_data = None
                        exc: BaseException | None = None
                        try:
                            ep_data = fut.result()
                        except BaseException as e:
                            exc = e

                        # Refill the pool BEFORE the (possibly blocking) put,
                        # so workers stay busy even if the writer is behind.
                        try:
                            next_p = next(pending_iter)
                            inflight[pool.submit(preprocess_episode, str(next_p))] = next_p
                        except StopIteration:
                            pass

                        result_queue.put((ep_path, ep_data, exc))
        except BaseException as e:
            producer_error.append(e)
        finally:
            result_queue.put(SENTINEL)

    producer_thread = threading.Thread(target=producer, name="g2-decoder", daemon=True)
    producer_thread.start()

    written = 0
    pbar = tqdm(total=len(h5_paths), desc="Preprocess+Write (pipelined)")
    while True:
        item = result_queue.get()
        if item is SENTINEL:
            break
        ep_path, ep_data, exc = item

        if exc is not None:
            print(f"Error preprocessing {ep_path}: {exc}")
        elif ep_data is not None:
            cam_frames = ep_data["cam_frames"]
            states = ep_data["states"]
            actions = ep_data["actions"]
            instruction = ep_data["language_instruction"]
            for t in range(ep_data["num_frames"]):
                dataset.add_frame({
                    "head_color": cam_frames["head_color"][t],
                    "hand_left_color": cam_frames["hand_left_color"][t],
                    "hand_right_color": cam_frames["hand_right_color"][t],
                    "state": states[t],
                    "actions": actions[t],
                    "task": instruction,
                })
            dataset.save_episode()
            written += 1
            del ep_data, cam_frames, states, actions

        pbar.update(1)
    pbar.close()

    producer_thread.join()
    if producer_error:
        raise producer_error[0]

    if hasattr(dataset, "consolidate"):
        dataset.consolidate()
    print(f"Dataset saved to {output_path}")
    print(f"Total episodes: {written}, Total frames: {dataset.num_frames}")

    if push_to_hub:
        dataset.push_to_hub(tags=["g2"], private=False, push_videos=True)


if __name__ == "__main__":
    tyro.cli(main)
