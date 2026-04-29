"""G2 dual-arm mobile robot policy transforms.

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
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

ACTION_DIM = 24
STATE_DIM = 26


def make_g2_example() -> dict:
    """Creates a random input example for the G2 policy."""
    return {
        "observation/head_color": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/hand_left_color": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/hand_right_color": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(STATE_DIM).astype(np.float32),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class G2Inputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)

        head_image = _parse_image(data["observation/head_color"])
        left_hand_image = _parse_image(data["observation/hand_left_color"])
        right_hand_image = _parse_image(data["observation/hand_right_color"])

        match self.model_type:
            case _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (head_image, left_hand_image, right_hand_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type for G2: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class G2Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM], dtype=np.float32)}
