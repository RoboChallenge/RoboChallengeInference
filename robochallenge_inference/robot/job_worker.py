import time
import logging
from pathlib import Path

import yaml

# Next to this file so running ``uv run python robochallenge_inference/demo.py`` from
# the repo root still works (no dependence on the process cwd).
_CONFIG_YAML = Path(__file__).resolve().parent / "config.yaml"


def process_job(client, gpu_client, job_id, robot_id,
                cameras, image_width, image_height, action_type, action_freq, target_objects,
                max_wait=600):
    """Process a single job: start robot, poll state, infer, post actions.

    Args:
        client: InterfaceClient instance.
        gpu_client: Inference client with an ``infer(state)`` method.
        job_id: Job identifier.
        robot_id: Robot identifier.
        cameras: Camera name list for ``client.get_state()``.
        image_width: Desired image width.
        image_height: Desired image height.
        action_type: ``"joint"`` or ``"pose"``.
        action_freq: Action trajectory frequency in Hz.
        max_wait: Timeout in seconds (default 600).
    """
    try:
        device, status, task_id, index = client.get_job_status(job_id)
        logging.info(f"Processing job_id: {job_id}, status: {status}")
        if status != "ready":
            return

        client.update_job_info(job_id, robot_id)
        r = client.start_robot(job_id)
        logging.info(f"Started robot: {r.content}")
        if r.status_code != 200:
            return

        start_time = time.time()
        while True:
            device, status, task_id, index = client.get_job_status(job_id)
            if status != "running":
                break

            playback = client.get_status()
            if playback and playback.get("status") != "free":
                time.sleep(0.5)
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
            actions = gpu_client.infer(
                state, target_objects[task_id][str(index)]
            )
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


def job_loop(client, gpu_client, run_id,
             cameras, image_width, image_height, action_type, action_freq):
    """Poll a job collection and process ready jobs until all are finished.

    Args:
        client: InterfaceClient instance.
        gpu_client: Inference client with an ``infer(state)`` method.
        run_id: Job collection / run identifier.
        cameras: Camera name list.
        image_width: Desired image width.
        image_height: Desired image height.
        action_type: ``"joint"`` or ``"pose"``.
        action_freq: Action trajectory frequency in Hz.
    """
    ACTIVE_STATES = ["assigned", "prepare", "ready", "running"]
    MAX_EMPTY_POLLS = 10
    empty_poll_count = 0

    with open(_CONFIG_YAML, encoding="utf-8") as f:
        target_objects = yaml.safe_load(f)

    while True:
        job_collection = client.get_all_jobs(run_id)
        jobs = job_collection["jobs"]

        has_active_job = False
        finished_count = 0
        for job in jobs:
            status = job["status"]
            if status in ACTIVE_STATES:
                has_active_job = True
                break
            elif status in ["finished", "cancelled", "failed"]:
                finished_count += 1

        if not has_active_job and finished_count == len(jobs):
            empty_poll_count += 1
            logging.info(f"No active jobs, poll count: {empty_poll_count}")
            if empty_poll_count >= MAX_EMPTY_POLLS:
                logging.info("No new jobs after multiple checks, exiting.")
                break
            time.sleep(1)
            continue
        else:
            empty_poll_count = 0

        for job in jobs:
            job_id = job["job_id"]
            robot_id = job["device"]["robot_id"]
            status = job["status"]
            logging.info(f"Job id: {job_id}, status: {status}, remaining jobs: {len(jobs)}")
            if status == "ready":
                process_job(
                    client, gpu_client, job_id, robot_id,
                    cameras, image_width, image_height,
                    action_type, action_freq, target_objects,
                )

        time.sleep(1)
