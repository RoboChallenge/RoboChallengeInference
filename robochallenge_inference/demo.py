"""Official RoboChallenge entry point: poll Arena jobs and run pi0.5 G2 inference."""

import argparse
import logging

from policy import G2_CAMERAS, Pi05Policy
from robot.interface_client import InterfaceClient
from robot.job_worker import job_loop

logging.basicConfig(
    filename='mylogfile.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)


class GPUClient:
    """Wraps a policy and turns ``target_objects`` into a language prompt."""

    def __init__(self, policy):
        self.policy = policy

    def _process_prompt(self, target_objects):
        """Build a prompt aligned with ``examples/g2/convert_g2_data_to_lerobot.py``.

        Training prompt format::

            single:  "Target: {a}. Pick the {a} from the shelf and place it into the cart."
            double:  "Target: {a} and {b}. Pick the {a} and the {b} from the shelf and place them into the cart."
        """
        items = [t for t in target_objects if t]
        if not items:
            return "Pick the item from the shelf and place it into the cart."

        if len(items) == 1:
            target_phrase = items[0]
            action_phrase = f"Pick the {items[0]} from the shelf and place it into the cart."
        else:
            target_phrase = " and ".join(items)
            action_object = " and ".join(f"the {it}" for it in items)
            action_phrase = f"Pick {action_object} from the shelf and place them into the cart."

        return f"Target: {target_phrase}. {action_phrase}"

    def infer(self, state, target_objects=("Pepsi",)):
        """Main entry point invoked by ``job_worker.process_job``."""
        prompt = self._process_prompt(list(target_objects))
        print(f"Prompt: {prompt}")
        return self.policy.run_policy(state, prompt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_token', type=str, required=True, help='User token (sent as x-user-id)')
    parser.add_argument('--run_id', type=str, required=True,
                        help='Job collection / run ID')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to a pi05_g2_finetune checkpoint directory')
    parser.add_argument('--train-config', dest='train_config', type=str,
                        default='pi05_g2_finetune_h50',
                        help='openpi training config name (must match the checkpoint)')
    parser.add_argument('--openpi-src', dest='openpi_src', type=str, default=None,
                        help='Optional path to <repo>/src, prepended to sys.path')
    parser.add_argument('--device', type=str, default=None,
                        help='PyTorch device (e.g. "cuda", "cuda:0"); ignored for JAX checkpoints')
    parser.add_argument('--action-freq', dest='action_freq', type=float, default=30.0,
                        help='Action trajectory sampling rate in Hz')

    args = parser.parse_args()

    cameras = list(G2_CAMERAS)
    image_width = Pi05Policy.IMG_SIZE
    image_height = Pi05Policy.IMG_SIZE
    action_type = "joint"

    client = InterfaceClient(args.user_token)
    policy = Pi05Policy(
        args.checkpoint,
        openpi_src=args.openpi_src,
        train_config_name=args.train_config,
        device=args.device,
    )
    gpu_client = GPUClient(policy)

    job_loop(
        client, gpu_client, args.run_id,
        cameras, image_width, image_height,
        action_type, args.action_freq,
    )


if __name__ == '__main__':
    main()
