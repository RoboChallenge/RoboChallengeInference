"""Local smoke test against ``mock_server`` — no Arena involvement.

Start ``mock_server/mock_server.py`` first, then::

    python3 test.py --checkpoint /path/to/pi05_g2_finetune_h50/<step>
"""

import argparse
import logging
import time

from demo import G2_CAMERAS, GPUClient
from policy import Pi05Policy
from robot.interface_client import InterfaceClient

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

DEFAULT_USER_TOKEN = "test_user"
DEFAULT_JOBS = ["test_job"]
DEFAULT_ROBOT_ID = "test_robot"
DEFAULT_TARGETS = ("Pepsi",)


def process_job(client, gpu_client, job_id, robot_id, target_objects,
                cameras, image_width, image_height, action_type, action_freq,
                max_wait=600):
    try:
        start_time = time.time()
        while True:
            playback = client.get_status()
            if playback and playback.get("status") != "free":
                time.sleep(0.02)
                continue

            state = client.get_state(
                cameras=cameras,
                image_width=image_width,
                image_height=image_height,
            )
            if not state:
                time.sleep(0.5)
                continue

            logging.info("get_state ok")
            actions = gpu_client.infer(state, target_objects)
            if actions:
                logging.info("Inference result: %d frames", len(actions))
                client.post_actions(
                    actions,
                    action_type=action_type,
                    action_freq=action_freq,
                )

            if time.time() - start_time > max_wait:
                logging.warning("Job %s exceeded max wait time.", job_id)
                break
    except Exception as e:
        logging.error("Error processing job %s: %s", job_id, e)
    finally:
        client.stop_motion()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to a pi05_g2_finetune checkpoint directory')
    parser.add_argument('--train-config', dest='train_config', type=str,
                        default='pi05_g2_finetune_h50',
                        help='openpi training config name (must match the checkpoint)')
    parser.add_argument('--openpi-src', dest='openpi_src', type=str, default=None,
                        help='Optional path to <repo>/src, prepended to sys.path')
    parser.add_argument('--device', type=str, default=None,
                        help='PyTorch device (e.g. "cuda", "cuda:0"); ignored for JAX checkpoints')
    parser.add_argument('--targets', type=str, nargs='+', default=list(DEFAULT_TARGETS),
                        help='Target object names used to build the prompt')
    parser.add_argument('--action-freq', dest='action_freq', type=float, default=30.0,
                        help='Action trajectory sampling rate in Hz')

    args = parser.parse_args()

    cameras = list(G2_CAMERAS)
    image_width = Pi05Policy.IMG_SIZE
    image_height = Pi05Policy.IMG_SIZE
    action_type = "joint"

    client = InterfaceClient(DEFAULT_USER_TOKEN, mock=True)
    client.update_job_info(DEFAULT_JOBS[0], DEFAULT_ROBOT_ID)

    policy = Pi05Policy(
        args.checkpoint,
        openpi_src=args.openpi_src,
        train_config_name=args.train_config,
        device=args.device,
    )
    gpu_client = GPUClient(policy)

    jobs = list(DEFAULT_JOBS)
    while jobs:
        for job_id in jobs[:]:
            try:
                process_job(
                    client, gpu_client, job_id, DEFAULT_ROBOT_ID, args.targets,
                    cameras, image_width, image_height,
                    action_type, args.action_freq,
                )
            except Exception as e:
                logging.error("Error processing job %s: %s", job_id, e)
            finally:
                jobs.remove(job_id)
    logging.info("All jobs processed.")
    return True


if __name__ == "__main__":
    main()
