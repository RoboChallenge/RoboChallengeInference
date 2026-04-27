import argparse
import logging
import pickle
import queue
import threading
import time
from pathlib import Path
from threading import Thread
from typing import List, Optional, Dict, Any

import cv2
import numpy as np
import uvicorn
from fastapi import APIRouter, FastAPI, Response, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from utils import resize_with_pad_single

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

cmd_Q = queue.Queue()
navi_Q = queue.Queue()
motion_lock = threading.Lock()
active_motion: Optional[str] = None  # "actions" / "navigating"

ARM_JOINT_COUNT = 14
HEAD_JOINT_COUNT = 3
WAIST_JOINT_COUNT = 5
GRIPPER_COUNT = 2

G2_COLOR_CAMERAS = ["kHeadColor", "kHandLeftColor", "kHandRightColor"]
G2_STEREO_CAMERAS = ["kHeadStereoLeft", "kHeadStereoRight"]
G2_DEPTH_CAMERAS = ["kHeadDepth"]
G2_ALL_CAMERAS = G2_COLOR_CAMERAS + G2_STEREO_CAMERAS + G2_DEPTH_CAMERAS

G2_IMG_FILE_MAP = {
    "kHeadColor": "kHeadColor.jpg",
    "kHandLeftColor": "kHandLeftColor.jpg",
    "kHandRightColor": "kHandRightColor.jpg",
    "kHeadStereoLeft": "kHeadStereoLeft.jpg",
    "kHeadStereoRight": "kHeadStereoRight.jpg",
    "kHeadDepth": "kHeadDepth.z16",
}

# Camera fixtures live next to this file (not CWD), so `python mock_server/mock_server.py` works from repo root.
_MOCK_ASSET_DIR = Path(__file__).resolve().parent

DEPTH_DEFAULT_W, DEPTH_DEFAULT_H = 640, 400


# ---------------------------------------------------------------------------
# Request models (mirrors arena_bridge.py)
# ---------------------------------------------------------------------------


class ClientRequest(BaseModel):
    client_id: str


class ActionRequest(ClientRequest):
    actions: List[Dict[str, Any]]
    action_type: str = "joint"
    action_freq: float = Field(default=30.0, gt=0.0, le=500.0)


class GotoNaviRequest(ClientRequest):
    position: Dict[str, Any]
    max_refine: int = Field(default=5, ge=1, le=100)


# ---------------------------------------------------------------------------
# Mock robot
# ---------------------------------------------------------------------------


class MockG2Robot:
    def __init__(self):
        self.arm_joints = [0.0] * ARM_JOINT_COUNT
        self.head_joints = [0.0] * HEAD_JOINT_COUNT
        self.waist_joints = [0.0] * WAIST_JOINT_COUNT
        self.gripper = [0.0] * GRIPPER_COUNT
        self.time_counter = 0
        self._color_imgs: Dict[str, np.ndarray] = {}
        self._depth_img: Optional[np.ndarray] = None
        for cam, filename in G2_IMG_FILE_MAP.items():
            asset_path = _MOCK_ASSET_DIR / filename
            if cam in G2_DEPTH_CAMERAS:
                self._depth_img = self._load_depth(asset_path)
            else:
                if asset_path.is_file():
                    img = cv2.imread(str(asset_path))
                    if img is not None:
                        self._color_imgs[cam] = img
                    else:
                        logger.warning("Could not decode mock image %s, using random RGB", asset_path)
                        self._color_imgs[cam] = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                else:
                    logger.debug("Mock camera asset missing %s, using random RGB", asset_path.name)
                    self._color_imgs[cam] = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    @staticmethod
    def _load_depth(path: Path) -> np.ndarray:
        """Load raw Z16 depth from file, or generate random fallback."""
        if not path.is_file():
            return np.random.randint(200, 5000, (DEPTH_DEFAULT_H, DEPTH_DEFAULT_W), dtype=np.uint16)
        try:
            raw = np.fromfile(str(path), dtype=np.uint16)
            if raw.size == DEPTH_DEFAULT_W * DEPTH_DEFAULT_H:
                return raw.reshape((DEPTH_DEFAULT_H, DEPTH_DEFAULT_W))
        except OSError:
            pass
        return np.random.randint(200, 5000, (DEPTH_DEFAULT_H, DEPTH_DEFAULT_W), dtype=np.uint16)

    def get_img(self, cam_name, image_size=None):
        src = self._color_imgs.get(cam_name)
        if src is None:
            src = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        if image_size is not None:
            src = cv2.resize(src, image_size)
        return src

    def get_depth(self) -> np.ndarray:
        return self._depth_img

    def get_enable(self):
        return True

    def get_joint(self):
        return (np.random.random(ARM_JOINT_COUNT).tolist() +
                np.random.random(HEAD_JOINT_COUNT).tolist() +
                np.random.random(WAIST_JOINT_COUNT).tolist())

    def get_pose(self):
        return np.random.random(ARM_JOINT_COUNT).tolist()

    def go_home(self):
        self.arm_joints = [0.0] * ARM_JOINT_COUNT
        self.head_joints = [0.0] * HEAD_JOINT_COUNT
        self.waist_joints = [0.0] * WAIST_JOINT_COUNT
        self.gripper = [0.0] * GRIPPER_COUNT

    def go_joint(self, joints):
        if len(joints) >= ARM_JOINT_COUNT:
            self.arm_joints = joints[:ARM_JOINT_COUNT]
        if len(joints) >= ARM_JOINT_COUNT + HEAD_JOINT_COUNT:
            self.head_joints = joints[ARM_JOINT_COUNT:ARM_JOINT_COUNT + HEAD_JOINT_COUNT]
        if len(joints) >= ARM_JOINT_COUNT + HEAD_JOINT_COUNT + WAIST_JOINT_COUNT:
            self.waist_joints = joints[ARM_JOINT_COUNT + HEAD_JOINT_COUNT:ARM_JOINT_COUNT + HEAD_JOINT_COUNT + WAIST_JOINT_COUNT]

    def go_pose(self, pose):
        pass


# ---------------------------------------------------------------------------
# FastAPI server (routes match arena_bridge.py)
# ---------------------------------------------------------------------------


class FlaskWorker(Thread):
    def __init__(self, server_port, robot: MockG2Robot):
        super().__init__()
        self.server_port = server_port
        self.robot = robot
        router = APIRouter()
        router.add_api_route("/clock_sync", self.clock_sync, methods=["GET"])
        router.add_api_route("/state", self.get_state, methods=["GET"])
        router.add_api_route("/actions", self.post_actions, methods=["POST"])
        router.add_api_route("/stop_motion", self.stop_motion, methods=["POST"])
        router.add_api_route("/status", self.get_status, methods=["GET"])
        router.add_api_route("/goto_navi_position", self.goto_navi_position, methods=["POST"])
        self.app = FastAPI(title="G2 Mock Robot Server")
        self.app.include_router(router)

    def clock_sync(self):
        ts = time.time()
        logger.info("GET /clock_sync → timestamp=%.3f", ts)
        return {"timestamp": ts}

    def get_state(
        self,
        client_id: str = "",
        cameras: Optional[List[str]] = Query(default=None),
        image_width: Optional[int] = Query(default=None),
        image_height: Optional[int] = Query(default=None),
    ):
        image_size = None
        if image_width is not None and image_height is not None:
            image_size = (image_width, image_height)

        cam_list = cameras if cameras else G2_ALL_CAMERAS

        camera_dict = {}
        ts_ns = int(time.time() * 1e9)
        for cam in cam_list:
            if cam not in G2_ALL_CAMERAS:
                continue
            if cam in G2_DEPTH_CAMERAS:
                depth = self.robot.get_depth()
                h, w = depth.shape[:2]
                camera_dict[cam] = {
                    "width": w,
                    "height": h,
                    "timestamp_ns": ts_ns,
                    "data": depth.tobytes(),
                    "encoding": "UNCOMPRESSED",
                    "color_format": "RS2_FORMAT_Z16",
                }
            else:
                src = self.robot.get_img(cam, image_size=image_size)
                h, w = src.shape[:2]
                img_bytes = cv2.imencode(".jpg", src, [int(cv2.IMWRITE_JPEG_QUALITY), 85])[-1].tobytes()
                camera_dict[cam] = {
                    "width": w,
                    "height": h,
                    "timestamp_ns": ts_ns,
                    "data": img_bytes,
                    "encoding": "JPEG",
                    "color_format": "RGB",
                }

        joint = self.robot.get_joint()
        pose = self.robot.get_pose()

        state_data = {
            "timestamp": ts_ns,
            "robot_position": {
                "arm_joint_position": joint[:ARM_JOINT_COUNT],
                "head_joint_position": joint[ARM_JOINT_COUNT:ARM_JOINT_COUNT + HEAD_JOINT_COUNT],
                "waist_joint_position": joint[ARM_JOINT_COUNT + HEAD_JOINT_COUNT:],
                "left_end_position": [0.0, 0.0, 0.0],
                "left_end_orientation": [0.0, 0.0, 0.0, 1.0],
                "right_end_position": [0.0, 0.0, 0.0],
                "right_end_orientation": [0.0, 0.0, 0.0, 1.0],
            },
            "camera": camera_dict,
            "gripper_position": [0.0, 0.0],
            "slam_pose": None,
        }
        payload = pickle.dumps(state_data, protocol=pickle.HIGHEST_PROTOCOL)
        cam_summary = {k: f"{v['width']}x{v['height']}" for k, v in camera_dict.items()}
        logger.info(
            "GET /state → client=%s cameras=%s image_size=%s payload=%d bytes",
            client_id, cam_summary, image_size, len(payload),
        )
        return Response(content=payload, media_type="application/vnd.python.pickle")

    def post_actions(self, request: ActionRequest):
        logger.info(
            "POST /actions → client=%s type=%s freq=%.1f frames=%d",
            request.client_id, request.action_type, request.action_freq, len(request.actions),
        )
        try:
            action_freq = request.action_freq
            period = 1.0 / action_freq if action_freq > 0 else 1.0 / 30.0
            for action_dict in request.actions:
                rp = action_dict.get("robot_position", {})
                if request.action_type == "joint":
                    joints = (
                        rp.get("arm_joint_position", [0.0] * ARM_JOINT_COUNT)
                        + rp.get("head_joint_position", [0.0] * HEAD_JOINT_COUNT)
                        + rp.get("waist_joint_position", [0.0] * WAIST_JOINT_COUNT)
                    )
                else:
                    joints = None
                if not cmd_Q.full():
                    cmd_Q.put({
                        "action": joints,
                        "action_dict": action_dict,
                        "duration": period,
                        "action_type": request.action_type,
                    })
                else:
                    logger.warning("POST /actions → queue full, rejected")
                    return JSONResponse(
                        {"status": "error", "message": "Queue full"},
                        status_code=429,
                    )
            return {"status": "accepted"}
        except Exception as e:
            logger.error("POST /actions → error: %s", e)
            return JSONResponse({"status": "error", "message": str(e)}, status_code=400)

    def stop_motion(self, request: ClientRequest):
        cleared = 0
        for q in (cmd_Q, navi_Q):
            while not q.empty():
                try:
                    q.get_nowait()
                    cleared += 1
                except queue.Empty:
                    break
        logger.info("POST /stop_motion → client=%s cleared=%d", request.client_id, cleared)
        return {"status": "success"}

    def get_status(self, client_id: str = ""):
        active = active_motion
        action_queue = cmd_Q.qsize()
        navigation_queue = navi_Q.qsize()
        resetting = False
        if resetting:
            status = "resetting"
        elif active == "navigating":
            status = "navigating"
        elif active == "actions":
            status = "pending actions"
        elif navigation_queue > 0 or action_queue > 0:
            status = "pending"
        else:
            status = "free"
        logger.info(
            "GET /status → client=%s active=%s action_queue=%d navi_queue=%d status=%s",
            client_id, active, action_queue, navigation_queue, status,
        )
        return {
            "action_queue": action_queue,
            "navigation_queue": navigation_queue,
            "resetting": resetting,
            "status": status,
        }

    def goto_navi_position(self, request: GotoNaviRequest):
        logger.info(
            "POST /goto_navi_position → client=%s position=%s max_refine=%d",
            request.client_id, request.position, request.max_refine,
        )
        navi_Q.put({
            "position": request.position,
            "max_refine": request.max_refine,
        })
        return {"status": "accepted"}

    def run(self):
        logger.info("Mock server starting on port %d", self.server_port)
        logger.info("Endpoints: GET /clock_sync, GET /state, POST /actions, "
                     "POST /stop_motion, GET /status, POST /goto_navi_position")
        uvicorn.run(self.app, host="0.0.0.0", port=self.server_port, access_log=False)


class ActionWorker(Thread):
    """Consumes cmd_Q; holds motion_lock during execution (blocks NaviWorker)."""

    def __init__(self, robot):
        super().__init__(daemon=True, name="action-worker")
        self.running = True
        self.robot = robot

    def run(self):
        while self.running:
            try:
                action_duration = cmd_Q.get(timeout=0.1)
            except queue.Empty:
                continue
            global active_motion
            t_cur = time.time()
            with motion_lock:
                active_motion = "actions"
                try:
                    action = action_duration["action"]
                    action_type = action_duration["action_type"]
                    if action is not None:
                        try:
                            if self.robot.get_enable():
                                if action_type == "pose":
                                    self.robot.go_pose(action)
                                else:
                                    self.robot.go_joint(action)
                        except Exception as e:
                            logger.error("Joint motion failed: %s", e)
                    period = action_duration["duration"]
                    elapsed = time.time() - t_cur
                    if elapsed < period:
                        time.sleep(period - elapsed)
                finally:
                    active_motion = None

    def stop(self):
        self.running = False


class NaviWorker(Thread):
    """Consumes navi_Q; holds motion_lock during execution (blocks ActionWorker)."""

    MOCK_NAVI_DURATION = 2.0

    def __init__(self):
        super().__init__(daemon=True, name="navi-worker")
        self.running = True

    def run(self):
        while self.running:
            try:
                task = navi_Q.get(timeout=0.1)
            except queue.Empty:
                continue
            global active_motion
            with motion_lock:
                active_motion = "navigating"
                try:
                    logger.info("NaviWorker → navigating to %s (lock acquired, actions blocked)",
                                task["position"])
                    time.sleep(self.MOCK_NAVI_DURATION)
                    logger.info("NaviWorker → navigation done (lock released)")
                finally:
                    active_motion = None

    def stop(self):
        self.running = False


class RobotDashboard:
    def __init__(self, server_port=9098):
        super().__init__()
        self.server_port = server_port
        self.robot = MockG2Robot()

        try:
            self.robot.go_home()
        except Exception as e:
            logging.error("Failed to reset robot: %s", e)

        time.sleep(2.0)

        flask_worker = FlaskWorker(self.server_port, self.robot)
        flask_worker.daemon = False
        flask_worker.start()
        ActionWorker(self.robot).start()
        NaviWorker().start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G2 Mock Robot Server")
    parser.add_argument("-s", "--server_port", type=int, default=9098)
    args = parser.parse_args()
    try:
        RobotDashboard(server_port=args.server_port)
    except KeyboardInterrupt:
        logger.info("Shutting down mock server (KeyboardInterrupt)")
    except Exception:
        logger.exception("Mock server failed to start")
        raise
