import requests
import numpy as np
import pickle
import time
from typing import List, Optional, Dict, Any
from utils.log import setup_logger
from utils.util import timeout
from utils.enums import ReturnCode
from utils.util import retry_request

logger = setup_logger()

base_url = "http://api.robochallenge.cn"
mock_url = "http://127.0.0.1:9098"

MAX_RETRY = 3
RETRY_DELAY = 1


class InterfaceClient:
    def __init__(self, user_token, mock=False):
        self.user_token = user_token
        self.session = requests.Session()
        self.session.headers.update({"x-user-id": user_token})
        self.job_id = None
        self.robot_id = None
        self.robot_url = None
        self.clock_offset = None
        self.mock = mock

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, url, **kwargs):
        @retry_request(retries=MAX_RETRY, delay=RETRY_DELAY)
        def inner():
            return self.session.get(url, **kwargs)
        return inner()

    def _post(self, url, **kwargs):
        @retry_request(retries=MAX_RETRY, delay=RETRY_DELAY)
        def inner():
            return self.session.post(url, **kwargs)
        return inner()

    def _put(self, url, **kwargs):
        @retry_request(retries=MAX_RETRY, delay=RETRY_DELAY)
        def inner():
            return self.session.put(url, **kwargs)
        return inner()

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------

    def update_job_info(self, job_id, robot_id):
        self.job_id = job_id
        self.robot_id = robot_id
        self.robot_url = base_url + f"/robots/{robot_id}/direct"
        if self.mock:
            self.robot_url = mock_url
        self.clock_offset = self._calibrate_clock_offset()
        logger.info("Clock offset: %.4fs", self.clock_offset)

    def reset_job_info(self):
        self.job_id = None
        self.robot_id = None
        self.robot_url = None
        self.clock_offset = None
        self.mock = False

    # ------------------------------------------------------------------
    # Robot API (6 core interfaces)
    # ------------------------------------------------------------------

    def clock_sync(self) -> Dict[str, Any]:
        """Get robot-side wall clock timestamp.

        Returns:
            dict with key ``timestamp`` (float, unix epoch seconds).
        """
        response = self._get(f"{self.robot_url}/clock_sync",)
        response.raise_for_status()
        return response.json()

    def get_state(
        self,
        cameras: Optional[List[str]] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get current robot observation (joint positions, camera images, gripper, slam pose).

        Args:
            cameras: Camera names to include. Allowed values:
                ``kHeadStereoLeft``, ``kHeadStereoRight``, ``kHandLeftColor``,
                ``kHandRightColor``, ``kHeadColor``, ``kHeadDepth``.
                Omit to get all regular cameras.
            image_width: Desired image width (must be set together with image_height).
                Only affects color cameras; depth cameras are never resized.
            image_height: Desired image height (must be set together with image_width).

        Returns:
            dict with keys ``timestamp``, ``robot_position``, ``camera``,
            ``gripper_position``, ``slam_pose``.
            Each camera entry contains ``width``, ``height``, ``timestamp_ns``,
            ``data`` (JPEG bytes for color, raw Z16 for depth),
            ``encoding`` ("JPEG" / "UNCOMPRESSED"), and ``color_format``
            ("RGB" / "RS2_FORMAT_Z16").
            Returns None on request failure.
        """
        params = {"client_id": self.job_id}
        if cameras is not None:
            params["cameras"] = cameras
        if image_width is not None:
            params["image_width"] = image_width
        if image_height is not None:
            params["image_height"] = image_height
        try:
            response = self._get(f"{self.robot_url}/state", params=params)
            response.raise_for_status()
            return pickle.loads(response.content)
        except requests.exceptions.RequestException as e:
            logger.error("Error getting state: %s", e)
            return None

    def post_actions(
        self,
        actions: List[Dict[str, Any]],
        action_type: str = "joint",
        action_freq: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        """Send a batch of target actions to the robot.

        The server resamples the trajectory from ``action_freq`` to the
        execution rate (100 Hz for joint, 50 Hz for pose).

        Args:
            actions: List of action dicts. Each dict may contain:

                ``robot_position`` (required) — joint or pose targets:
                    joint mode::

                        {"arm_joint_position": [float]*14,
                         "head_joint_position": [float]*3,
                         "waist_joint_position": [float]*5}

                    pose mode::

                        {"left_end_position": [x,y,z],
                         "left_end_orientation": [qx,qy,qz,qw],
                         "right_end_position": [x,y,z],
                         "right_end_orientation": [qx,qy,qz,qw]}

                ``gripper_position`` (optional) — ``[left, right]`` gripper values.

                ``chassis_velocity`` (optional) — ``[vx, vy, wz]`` chassis velocity.

            action_type: ``"joint"`` or ``"pose"``.
            action_freq: Source trajectory sampling rate in Hz (default 30).

        Returns:
            dict with ``status`` key, or None on failure.
        """
        data = {
            "client_id": self.job_id,
            "actions": actions,
            "action_type": action_type,
            "action_freq": action_freq,
        }
        try:
            response = self._post(f"{self.robot_url}/actions", json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error("Error posting actions: %s", e)
            return None

    def stop_motion(self) -> Optional[Dict[str, Any]]:
        """Cancel all pending actions on the robot.

        Returns:
            dict with ``status`` key, or None on failure.
        """
        data = {"client_id": self.job_id}
        try:
            response = self._post(f"{self.robot_url}/stop_motion", json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error("Error stopping motion: %s", e)
            return None

    def get_status(self) -> Optional[Dict[str, Any]]:
        """Get motion playback status.

        Returns:
            dict with keys ``action_queue``, ``navigation_queue``, ``resetting``, ``status``.

            ``status``: ``"free"`` / ``"resetting"`` / ``"navigating"`` /
            ``"pending actions"`` / ``"pending"``.

            Returns None on failure.
        """
        params = {"client_id": self.job_id}
        try:
            response = self._get(f"{self.robot_url}/status", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error("Error getting status: %s", e)
            return None

    def goto_navi_position(
        self,
        position: Dict[str, Any],
        max_refine: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """Navigate the robot chassis to a map-frame goal.

        Args:
            position: Goal dict with keys ``position`` [x, y, z] in metres
                and ``orientation`` [qx, qy, qz, qw] quaternion.
            max_refine: Max chassis refinement iterations (1–100, default 5).

        Returns:
            dict with ``status`` key, or None on failure.
        """
        data = {
            "client_id": self.job_id,
            "position": position,
            "max_refine": max_refine,
        }
        try:
            response = self._post(f"{self.robot_url}/goto_navi_position", json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error("Error in goto_navi_position: %s", e)
            return None

    # ------------------------------------------------------------------
    # Clock calibration (internal)
    # ------------------------------------------------------------------

    def _calibrate_clock_offset(self, samples: int = 10) -> float:
        offsets = []
        while True:
            try:
                for _ in range(samples):
                    t1 = time.time()
                    data = self.clock_sync()
                    t2 = float(data["timestamp"])
                    t3 = time.time()
                    offsets.append(((t2 - t1) + (t2 - t3)) / 2)
                    time.sleep(0.5)
                break
            except requests.exceptions.RequestException as e:
                logger.warning("Clock sync retry: %s", e)
                time.sleep(0.5)
        return float(np.array(offsets).mean())

    # ------------------------------------------------------------------
    # Arena platform APIs (job / batch management)
    # ------------------------------------------------------------------

    def start_robot(self, job_id):
        url = f"{base_url}/jobs/update"
        response = self._post(url, json={"job_id": job_id, "action": "start"}, headers={"x-user-id": self.user_token})
        return response

    def list_jobs(self):
        response = self._get(f"{base_url}/jobs", headers={"x-user-id": self.user_token})
        return response.json()

    def create_job(self, task_id, job_name):
        response = self._post(f"{base_url}/jobs", headers={"x-user-id": self.user_token}, json={"task_id": task_id, "job_name": job_name})
        return response.json()

    def create_batch_jobs(self, task_id, times, model, job_collection_name):
        data = {"model": model, "times": times, "task_id": task_id, "job_collection_name": job_collection_name}
        response = self._post(f"{base_url}/job_collections", json=data, headers={"x-user-id": self.user_token})
        return response.json()

    def get_batch_status(self, batch_id):
        response = self._get(f"{base_url}/job_collections/{batch_id}", headers={"x-user-id": self.user_token}).json()
        device = response.get("device")
        cloud_info = response.get("cloud_info")
        return {
            "status": response["status"],
            "robot_id": device.get("robot_id") if device else None,
            "task_id": response["task"]["task_id"],
            "cloud_info": cloud_info,
        }

    def put_batch_cloud_info(self, batch_id, cloud_info):
        response = self._put(f"{base_url}/job_collections/cloud/{batch_id}", headers={"x-user-id": self.user_token}, json=cloud_info)
        return response.json()

    def get_all_batch_status(self):
        response = self._get(f"{base_url}/job_collections", headers={"x-user-id": self.user_token}).json()
        for batch in response["jobs"]:
            print(batch)
        return response

    def _get_job_status(self, job_id):
        response = self._get(f"{base_url}/jobs/{job_id}", headers={"x-user-id": self.user_token})
        return response.json()

    def wait_for_robot_ready(self, job_id, poll_interval=2):
        while True:
            res = self._get_job_status(job_id)
            if "device" in res and "robot_id" in res:
                robot_id = res["device"]["robot_id"]
                return robot_id, job_id
            time.sleep(poll_interval)

    @timeout(600)
    def wait_for_robot_running(self, job_id, poll_interval=2):
        while True:
            res = self._get_job_status(job_id)
            if res and "status" in res:
                if res["status"] == "running":
                    return ReturnCode.SUCCESS
                elif res["status"] == "prepare":
                    pass
                else:
                    return ReturnCode.FAILURE
            time.sleep(poll_interval)

    def get_job_status(self, job_id):
        response = self._get_job_status(job_id)
        return response["device"], response["status"], response["task_id"], response["index"]

    def get_tasks(self):
        response = self._get(f"{base_url}/tasks", headers={"x-user-id": self.user_token})
        return response.json()

    def get_all_jobs(self, run_id):
        response = self._get(f"{base_url}/job_collections/{run_id}", headers={"x-user-id": self.user_token})
        return response.json()
