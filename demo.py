import argparse
import logging

from robot.interface_client import InterfaceClient
from robot.job_worker import job_loop

logging.basicConfig(
    filename='mylogfile.log',  # Log file name
    level=logging.INFO,  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s %(levelname)s:%(message)s'  # Log format
)


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

    def run_policy(self, input_data):
        """
        Run inference using the policy/model.
        Args:
            input_data: Input data for inference.
        Returns:
            list: Inference results.
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

    def infer(self, state):
        """
        Main entry point for inference.
        Args:
            state: Input state for the policy. Refer to README.md#get-state response example for details. It's unpickled and passed as a dict here.
        Returns:
            list: Inference results from the policy. Refer to README.md#post-action request parameters for details. This will be the `actions` field in the request.
        """
        result = self.policy.run_policy(state)
        return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--user_token', type=str, required=True, help='User token')
    parser.add_argument('--job_collection_id', type=str, required=True, help='Job collection id')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    # you can add your own parameters here

    args = parser.parse_args()

    # these args are generally not changed during evaluation, so we put them here.
    image_size = [224, 224] # this refers to README.md#get-state request parameter `width` and `height`
    image_type = ["high", "left_hand", "right_hand"] # this refers to README.md#get-state request parameter `image_type`
    action_type = "joint" # this refers to both README.md#get-state and README.md#post-action parameters `action_type`
    duration = 0.05 # this refers to README.md#post-action request parameter `duration`

    client = InterfaceClient(args.user_token)
    policy = DummyPolicy(args.checkpoint)  # add your own parameters
    gpu_client = GPUClient(policy)  # add your own parameters

    # main job loop. This function monitors when jobs are ready to eval and do the evaluation
    job_loop(client, gpu_client, args.job_collection_id, image_size, image_type, action_type, duration)


if __name__ == '__main__':
    main()
