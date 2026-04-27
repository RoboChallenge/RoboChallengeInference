import argparse
import logging
import time

from demo import GPUClient, DummyPolicy, G2_CAMERAS
from robot.interface_client import InterfaceClient

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

DEFAULT_USER_ID = "test_user"
DEFAULT_JOBS = ["test_job"]
DEFAULT_ROBOT_ID = "test_robot"


def process_job(client, gpu_client, job_id, robot_id,
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
            actions = gpu_client.infer(state)
            if actions:
                logging.info(f"Inference result: {len(actions)} frames")
                client.post_actions(
                    actions,
                    action_type=action_type,
                    action_freq=action_freq,
                )

            if time.time() - start_time > max_wait:
                logging.warning(f"Job {job_id} exceeded max wait time.")
                break
    except Exception as e:
        logging.error(f"Error processing job {job_id}: {e}")
    finally:
        client.stop_motion()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')

    args = parser.parse_args()

    cameras = G2_CAMERAS
    image_width = 224
    image_height = 224
    action_type = "joint"
    action_freq = 30.0

    client = InterfaceClient(DEFAULT_USER_ID, mock=True)
    client.update_job_info(DEFAULT_JOBS[0], DEFAULT_ROBOT_ID)

    policy = DummyPolicy(args.checkpoint)
    gpu_client = GPUClient(policy)

    jobs = DEFAULT_JOBS

    while jobs:
        for job_id in jobs[:]:
            try:
                process_job(
                    client, gpu_client, job_id, DEFAULT_ROBOT_ID,
                    cameras, image_width, image_height,
                    action_type, action_freq,
                )
                jobs.remove(job_id)
            except Exception as e:
                logging.error(f"Error processing job {job_id}: {e}")
                jobs.remove(job_id)
    logging.info("All jobs processed.")
    return True


if __name__ == "__main__":
    main()
