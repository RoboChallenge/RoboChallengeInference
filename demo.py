import argparse
import logging

import cv2
import numpy as np

from robot.interface_client import InterfaceClient
from robot.job_worker import job_loop
from openpi.training import config as _config
from openpi.policies import policy_config

logging.basicConfig(
    filename='mylogfile.log',  # Log file name
    level=logging.INFO,  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s %(levelname)s:%(message)s'  # Log format
)


class GPUClient:
    """
    Inference client class.
    """

    def __init__(self, checkpoint_path):
        """
        Initialize the inference client with a policy.
        Args:
            policy (DummyPolicy): An instance of the policy class.
        """
        config = _config.get_config("pi05_ft_aloha_qpos_stackbowls_fps15")

        # Create a trained policy.
        policy = policy_config.create_trained_policy(config,checkpoint_path)
        self.policy = policy
        self.image_mapping = {"high": "observation/cam_high",
                              "left_hand": "observation/cam_wrist_left",
                              "right_hand": "observation/cam_wrist_right",
                              }

    def decode(self, b: bytes):
        image = cv2.imdecode(
            np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def infer(self, state):
        """
        Main entry point for inference.
        Args:
            state: Input state for the policy. Refer to README.md#get-state response example for details. It's unpickled and passed as a dict here.
        Returns:
            list: Inference results from the policy. Refer to README.md#post-action request parameters for details. This will be the `actions` field in the request.
        """
        inputs = {}
        for k in self.image_mapping:
            inputs[self.image_mapping[k]] = self.decode(state['images'][k])
        inputs['prompt'] = 'stack_the_bowls_together'
        action = np.array(state['action'], np.float32)
        inputs['state'] = action[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]]
        action_chunk = self.policy.infer(inputs)["actions"][:16]
        actions = np.array(action_chunk)
        result = actions[:, [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]]
        return result.tolist()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--user_token', type=str, required=True, help='User token')
    parser.add_argument('--run_id', type=str, required=True, help='Run ID. Get it from the detail page of your submission')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    # you can modify or add your own parameters

    args = parser.parse_args()

    # these args are generally not changed during evaluation, so we put them here.
    image_size = [224, 224]  # this refers to README.md#get-state request parameter `width` and `height`
    image_type = ["high", "left_hand", "right_hand"]  # this refers to README.md#get-state request parameter `image_type`
    action_type = "joint"  # this refers to both README.md#get-state and README.md#post-action parameters `action_type`
    duration = 0.05  # this refers to README.md#post-action request parameter `duration`

    client = InterfaceClient(args.user_token)
    gpu_client = GPUClient(args.checkpoint)  # add your own parameters

    # main job loop. This function monitors when jobs are ready to eval and do the evaluation
    job_loop(client, gpu_client, args.run_id, image_size, image_type, action_type, duration)


if __name__ == '__main__':
    main()
