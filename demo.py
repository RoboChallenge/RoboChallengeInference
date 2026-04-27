import argparse
import logging

from robot.interface_client import InterfaceClient
from robot.job_worker import job_loop

logging.basicConfig(
    filename='mylogfile.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

G2_CAMERAS = ["kHeadColor", "kHandLeftColor", "kHandRightColor"]


class DummyPolicy:
    """
    Example policy class.
    Users should implement the __init__ and run_policy methods according to their own logic.
    """
    def __init__(self, checkpoint_path):
        """
        Initialize the policy.
        Args:
            checkpoint_path (str): Path to the model checkpoint file.
        """
        pass  # TODO: Load your model here using the checkpoint_path

    def run_policy(self, state, prompt):
        """
        Run inference using the policy/model.
        Args:
            state: Input state dict from client.get_state().
                The state dict has the following structure:
                {
                    "timestamp": int,                            # state capture time (nanoseconds)
                    "robot_position": {
                        "arm_joint_position": list[float],       # (14,) rad
                        "head_joint_position": list[float],      # (3,)  rad
                        "waist_joint_position": list[float],     # (5,)  rad
                        "left_end_position": list[float],        # (3,)  metres, base_link frame
                        "left_end_orientation": list[float],     # (4,)  quaternion [x,y,z,w]
                        "right_end_position": list[float],
                        "right_end_orientation": list[float],
                    },
                    "camera": {
                        "<camera_name>": {
                            "width": int,
                            "height": int,
                            "timestamp_ns": int,
                            "data": bytes,           # JPEG for color, raw Z16 for depth
                            "encoding": str,         # "JPEG" or "UNCOMPRESSED"
                            "color_format": str,     # "RGB" or "RS2_FORMAT_Z16"
                        },
                        ...
                    },
                    "gripper_position": list[float],  # (2,) [left, right]
                    "slam_pose": dict or None,        # {"position": [x,y,z], "orientation": [x,y,z,w]}
                }
            prompt: Prompt for the policy.
        Returns:
            list[dict]: List of action dicts. Each dict should contain:
                - "robot_position": dict with joint or pose targets
                    joint mode: {"arm_joint_position": [float]*14,
                                 "head_joint_position": [float]*3,
                                 "waist_joint_position": [float]*5}
                    pose mode:  {"left_end_position": [x,y,z],
                                 "left_end_orientation": [qx,qy,qz,qw],
                                 "right_end_position": [x,y,z],
                                 "right_end_orientation": [qx,qy,qz,qw]}
                - "gripper_position": [left, right] (optional)
                - "chassis_velocity": [vx, vy, wz] (optional)
        """
        # TODO: Implement your inference logic here (e.g., GPU model inference)
        return []


class GPUClient:
    """
    Inference client class.
    """

    def __init__(self, policy):
        """
        Initialize the inference client with a policy.
        Args:
            policy (DummyPolicy): An instance of the policy class.
        """
        self.policy = policy
    
    def _process_prompt(self, target_objects):
        """
        Process the target objects into a prompt.
        Args:
            target_objects: List of target objects.
        Returns:
            str: Prompt.
        """
        if len(target_objects) == 1:
            prompt = (
                "Move in front of the shelf. "
                "Grasp the {}. "
                "Hold the item and walk to the cart. "
                "Place the item into the cart."
            ).format(target_objects[0])
        else:
            prompt = (
                "Move in front of the shelf. "
                "Grasp the {} and the {}. "
                "Hold the items and walk to the cart. "
                "Place the items into the cart."
            ).format(target_objects[0], target_objects[1])
        return prompt

    def infer(self, state, target_objects=["Pepsi"]):
        """
        Main entry point for inference.
        Args:
            state: Input state dict from client.get_state().
        Returns:
            list[dict]: Action trajectory from the policy.
        """

        result = self.policy.run_policy(state, self._process_prompt(target_objects))
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_id', type=str, required=True, help='User ID')
    parser.add_argument('--job_collection_id', type=str, required=True,
                        help='Job collection / run ID')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint path')

    args = parser.parse_args()

    cameras = G2_CAMERAS
    image_width = 224
    image_height = 224
    action_type = "joint"
    action_freq = 30.0

    client = InterfaceClient(args.user_id)
    policy = DummyPolicy(args.checkpoint)
    gpu_client = GPUClient(policy)

    job_loop(
        client, gpu_client, args.job_collection_id,
        cameras, image_width, image_height,
        action_type, action_freq,
    )


if __name__ == '__main__':
    main()
