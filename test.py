import argparse
import logging
import time

from demo import GPUClient, OpenpiPolicy
from robot.interface_client import InterfaceClient

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

DEFAULT_USER_ID = "test_user"
DEFAULT_JOBS = ["test_job"]
DEFAULT_ROBOT_ID = "test_robot"

def process_job(client, gpu_client, job_id, robot_id, image_size, image_type, action_type, duration, max_wait=600):
    try:
        start_time = time.time()
        while True:
            client.start_motion()
            logging.info("Started robot")
            state = client.get_state(image_size, image_type, action_type)
            if not state:
                time.sleep(0.5)
                continue
            if state['state'] == "size_none":
                client.post_size()
                time.sleep(0.5)
                continue
            if state['state'] != "normal" or state['pending_actions'] != 0:
                time.sleep(0.5)
                continue
            logging.info("get_robot_state time: %.2f", time.time() - state['timestamp'])
            result = gpu_client.infer(state)
            logging.info(f"Inference result: {result}")
            # If you are unsure about the structure of the action (for example, its shape), you can refer to the `action` field of the `get_status` response.
            # For more information, please refer to the README.md file https://github.com/RoboChallenge/RoboChallengeInference?tab=readme-ov-file#robot-specific-notes.
            client.post_actions(result, duration, action_type)
            if time.time() - start_time > max_wait:
                logging.warning(f"Job {job_id} exceeded max wait time.")
                break
        client.end_motion()
    except Exception as e:
        logging.error(f"Error processing job {job_id}: {e}")
    finally:
        client.end_motion()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    parser.add_argument("--prompt", type=str, required=True, help="Prompt")
    parser.add_argument("--action_type", type=str, default="joint", help="Action type")
    parser.add_argument("--duration", type=float, default=0.033, help="Duration")
    parser.add_argument("--valid_action_num", type=int, default=30, help="Valid action num")
    parser.add_argument("--image_size", type=str, default="640x480", help="Image shape")
    parser.add_argument(
        "--robot_type",
        type=str,
        default="arx5",
        help="Robot type",
        choices=["arx5", "aloha", "ur5", "dosw"],
    )

    args = parser.parse_args()
    
    image_size = [int(dim) for dim in args.image_size.split("x")]
    # this refers to README.md#get-state request parameter `image_type`
    if args.robot_type in ["aloha", "dosw"]:
        image_type = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    elif args.robot_type == "arx5":
        image_type = ["cam_global", "cam_arm", "cam_side"]
    elif args.robot_type == "ur5":
        image_type =["cam_global", "cam_arm"]
    action_type = args.action_type  # this refers to both README.md#get-state and README.md#post-action parameters `action_type`
    duration = args.duration  # this refers to README.md#post-action request parameter `duration`
    valid_action_num = args.valid_action_num
    robot_type = args.robot_type

    client = InterfaceClient(DEFAULT_USER_ID,mock=True)
    client.update_job_info(DEFAULT_JOBS[0], DEFAULT_ROBOT_ID)
    
    if args.robot_type == "dosw":
        config_name = "cvpr_multitask_dosw1_rtc"
    elif args.robot_type == "aloha":
        config_name = "cvpr_multitask_aloha_rtc"
    elif args.robot_type == "arx5":
        config_name = "cvpr_multitask_arx5_rtc"
    elif args.robot_type == "ur5":
        config_name = "cvpr_multitask_ur5_rtc"
    policy = OpenpiPolicy(config_name, args.checkpoint, args.prompt, valid_action_num=valid_action_num, duration=duration, action_type=action_type, image_size=image_size, robot_type=robot_type)
    
    gpu_client = GPUClient(policy)

    jobs = DEFAULT_JOBS

    while jobs:
        for job_id in jobs[:]:
            try:
                process_job(
                    client, gpu_client, job_id, DEFAULT_ROBOT_ID,
                    image_size, image_type, action_type, duration
                )
                jobs.remove(job_id)
            except Exception as e:
                logging.error(f"Error processing job {job_id}: {e}")
                jobs.remove(job_id)
    logging.info("All jobs processed.")
    return True

if __name__ == "__main__":
    main()
