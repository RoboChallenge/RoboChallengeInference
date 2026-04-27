# RoboChallengeInference — G2

Offline inference code for the **G2** dual-arm humanoid: poll Arena jobs, read robot state, run your policy, and stream actions back to the real robot (or a local mock).

## Project Structure

```
RoboChallengeInference-G2/
├── README.md
├── requirements.txt
├── demo.py                 # Official evaluation: Arena job_loop + your policy
├── test.py                 # Local loop against mock (no job collection)
├── robot/
│   ├── config.yaml         # task_id → slot index → target object name(s)
│   ├── __init__.py
│   ├── interface_client.py # HTTP client to Arena + robot
│   └── job_worker.py       # job_loop, process_job
├── mock_server/
│   ├── mock_server.py      # FastAPI mock of robot direct API
│   └── utils.py
└── utils/
    ├── __init__.py
    ├── enums.py
    ├── log.py
    └── util.py
```

## User Guide

### 1. Installation

```bash
git clone <this-repository-url>
cd RoboChallengeInference-G2

# (Recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Implement your policy

- Edit `demo.py`: replace `DummyPolicy` with your model (load weights in `__init__`, run inference in `run_policy(state, prompt)`).
- `GPUClient` builds a language prompt from `target_objects` (see `robot/config.yaml` and `_process_prompt`).
- Default cameras are `G2_CAMERAS` in `demo.py` (`kHeadColor`, `kHandLeftColor`, `kHandRightColor`); you can change resolution via `job_loop` arguments.

### 3. Target mapping (`robot/config.yaml`)

`job_loop` loads `robot/config.yaml` at startup. The Arena job API returns a **`task_id`** and a per-task **`index`**. The YAML file maps them to the **list of product names** passed to `GPUClient.infer()`.

- Top-level keys = `task_id` values returned by the platform (e.g. `22er`, `22et`).
- Second-level keys = **string** indices `"0"` … `"9"` (JSON may send integers; the code uses `str(index)` when looking up).
- Values = a list of one or two product name strings, matching the task type (single or dual object).

**Run processes from the repository root** so the path `robot/config.yaml` resolves. Keep task keys in sync with the tasks you register on the platform.

### 4. Local test (mock)

```bash
# Terminal 1: mock robot API (default: http://127.0.0.1:9098 when using mock in InterfaceClient)
python3 mock_server/mock_server.py

# Terminal 2: run test client (see test.py; uses InterfaceClient in mock mode)
python3 test.py --checkpoint /path/to/your/checkpoint
```

Use this to debug observation decoding and action posting without an Arena run.

### 5. Submit

- Log in to RoboChallenge Web and submit an evaluation request.
- On **My Submissions**, open **Detail** for your run and copy the **Run ID** (used as `job_collection_id` below).

### 6. Execute (official `demo.py`)

When your run is scheduled, start the client **from the repo root** and keep it online for the session:

```bash
python3 demo.py \
  --user_id <your_robochallenge_user_id> \
  --job_collection_id <Run_ID_from_My_Submissions> \
  --checkpoint /path/to/your/checkpoint
```

- Logs are written to `mylogfile.log` (see `demo.py` logging config).
- The process polls the job collection, connects to the assigned robot, and exits after jobs finish (see `job_loop` in `robot/job_worker.py`).

If anything fails, check logs and that `robot/config.yaml` contains entries for the job’s `task_id` and index.

### 7. Result

When execution finishes, open **My Submissions** on the website to view scores and details.

## Key API Parameter Descriptions

The **production** base host is `http://api.robochallenge.cn`. After `update_job_info(job_id, robot_id)` is called, the robot’s direct control base path is:

`http://api.robochallenge.cn/robots/<robot_id>/direct`

For example, state is `.../direct/state?client_id=<job_id>`. With `InterfaceClient(..., mock=True)`, requests go to `http://127.0.0.1:9098` (same path shape), e.g. after starting `mock_server/mock_server.py` on the default port.

All robot interactions go through `InterfaceClient` (`robot/interface_client.py`). `job_loop` sets `update_job_info` from the job list — the **`robot_id` comes from the platform** (`job["device"]["robot_id"]`); you do not hard-code it in `demo.py`.

```python
from robot.interface_client import InterfaceClient
from robot.job_worker import job_loop

client = InterfaceClient(user_id="your_user_id")

# job_loop automatically polls the Arena platform for assigned jobs,
# extracts robot_id from the job info, and calls client.update_job_info() internally.
job_loop(client, gpu_client, job_collection_id,
         cameras=["kHeadColor", "kHandLeftColor", "kHandRightColor"],
         image_width=224, image_height=224,
         action_type="joint", action_freq=30.0)
```

---

### Sync Clock

**Endpoint:** `/clock_sync`
**Method:** `GET`

#### Request Parameters

None

#### Response Example

```json
{
  "timestamp": 1743400054.123
}
```

#### Response Fields

| Field     | Type  | Description                 |
|-----------|-------|-----------------------------|
| timestamp | float | Unix timestamp on the robot |

---

### Get State

**Endpoint:** `/state`
**Method:** `GET`

#### Request Parameters

| Parameter    | Type        | Required | Default             | Description                                                                                                          |
|--------------|-------------|----------|---------------------|----------------------------------------------------------------------------------------------------------------------|
| cameras      | list of str | No       | all regular cameras | Camera names to include. Allowed: `kHeadStereoLeft`, `kHeadStereoRight`, `kHandLeftColor`, `kHandRightColor`, `kHeadColor`, `kHeadDepth` |
| image_width  | int         | No*      | native resolution   | Width of the returned color images (1–8192). Must be set together with `image_height`                                |
| image_height | int         | No*      | native resolution   | Height of the returned color images (1–8192). Must be set together with `image_width`                                |

> \* `image_width` and `image_height` must both be set or both omitted.
>
> If the requested size exceeds the camera's native resolution, the original image is returned without upscaling.
> Depth cameras (`kHeadDepth`) are never resized regardless of `image_width`/`image_height`.

Additional notes on camera positions:

- `kHeadColor` — RGB camera on the robot head (640 × 400)
- `kHeadDepth` — Depth camera on the robot head (640 × 400, raw Z16)
- `kHandLeftColor` — RGB camera on the left arm (1280 × 1056)
- `kHandRightColor` — RGB camera on the right arm (1280 × 1056)
- `kHeadStereoLeft` / `kHeadStereoRight` — Stereo cameras on the head

#### Response Example

The response is a pickle file containing a dictionary with the following structure:

```python
{
    "timestamp": 1743400054123456789,               # int, state capture time (nanoseconds)
    "robot_position": {
        "arm_joint_position": [0.0, ...],           # list[float] (14,) arm motor positions (rad)
        "head_joint_position": [0.0, ...],          # list[float] (3,)  head motor positions (rad)
        "waist_joint_position": [0.0, ...],         # list[float] (5,)  waist motor positions (rad)
        "left_end_position": [x, y, z],             # list[float] (3,) metres, base_link frame
        "left_end_orientation": [qx, qy, qz, qw],  # list[float] (4,) quaternion
        "right_end_position": [x, y, z],
        "right_end_orientation": [qx, qy, qz, qw],
    },
    "camera": {
        "kHeadColor": {                             # color camera
            "width": 640,
            "height": 400,
            "timestamp_ns": 1743400054123456789,
            "data": b'\xff\xd8...',                 # JPEG bytes
            "encoding": "JPEG",
            "color_format": "RGB",
        },
        "kHeadDepth": {                             # depth camera
            "width": 640,
            "height": 400,
            "timestamp_ns": 1743400054123456789,
            "data": b'\x00\x00...',                 # raw Z16 bytes (uint16 LE, millimetres)
            "encoding": "UNCOMPRESSED",
            "color_format": "RS2_FORMAT_Z16",
        },
        # ... other requested cameras ...
    },
    "gripper_position": [0.0, 0.0],                 # list[float] (2,) [left, right]
    "slam_pose": {                                   # None if SLAM unavailable
        "position": [x, y, z],
        "orientation": [qx, qy, qz, qw],
    }
}
```

#### Response Fields

| Field                                    | Type         | Description                                                    |
|------------------------------------------|--------------|----------------------------------------------------------------|
| timestamp                                | int          | State capture timestamp in nanoseconds                         |
| robot_position.arm_joint_position        | list[float]  | (14,) arm motor positions in radians                           |
| robot_position.head_joint_position       | list[float]  | (3,) head motor positions in radians                           |
| robot_position.waist_joint_position      | list[float]  | (5,) waist motor positions in radians                          |
| robot_position.left_end_position         | list[float]  | (3,) left end-effector [x, y, z] in metres (base_link frame)  |
| robot_position.left_end_orientation      | list[float]  | (4,) left end-effector quaternion [x, y, z, w]                 |
| robot_position.right_end_position        | list[float]  | (3,) right end-effector [x, y, z] in metres (base_link frame) |
| robot_position.right_end_orientation     | list[float]  | (4,) right end-effector quaternion [x, y, z, w]                |
| camera.\<name\>.width                   | int          | Image width in pixels                                          |
| camera.\<name\>.height                  | int          | Image height in pixels                                         |
| camera.\<name\>.timestamp_ns            | int          | Per-camera capture timestamp in nanoseconds                    |
| camera.\<name\>.data                    | bytes        | Image data (JPEG for color, raw Z16 for depth)                 |
| camera.\<name\>.encoding                | str          | `"JPEG"` for color, `"UNCOMPRESSED"` for depth                 |
| camera.\<name\>.color_format            | str          | `"RGB"` for color, `"RS2_FORMAT_Z16"` for depth                |
| gripper_position                         | list[float]  | (2,) [left, right] gripper positions                           |
| slam_pose                               | dict or None | SLAM pose; None if unavailable                                 |

---

### Post Action

**Endpoint:** `/actions`
**Method:** `POST`

Send a batch of target actions to the robot. The server automatically resamples the trajectory from `action_freq` to the execution rate (100 Hz for joint mode, 50 Hz for pose mode).

Each action is a **dict** containing `robot_position` (required), and optionally `gripper_position` and `chassis_velocity`.

#### Request Parameters

| Parameter   | Type       | Required | Default  | Description                                                                      |
|-------------|------------|----------|----------|----------------------------------------------------------------------------------|
| actions     | list[dict] | Yes      | —        | List of action dicts (see format below). Each dict is one frame of the trajectory |
| action_type | str        | No       | "joint"  | Control mode: `"joint"` or `"pose"`                                              |
| action_freq | float      | No       | 30.0     | Source trajectory sampling rate in Hz (0 < freq ≤ 500)                           |

The HTTP body should be a JSON object with the following structure:

**Joint mode example:**

```json
{
  "actions": [
    {
      "robot_position": {
        "arm_joint_position": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "head_joint_position": [0.0, 0.0, 0.0],
        "waist_joint_position": [0.0, 0.0, 0.0, 0.0, 0.0]
      },
      "gripper_position": [0.0, 0.0],
      "chassis_velocity": [0.0, 0.0, 0.0]
    }
  ],
  "action_type": "joint",
  "action_freq": 30.0
}
```

**Pose mode example:**

```json
{
  "actions": [
    {
      "robot_position": {
        "left_end_position": [0.3, 0.2, 0.5],
        "left_end_orientation": [0.0, 0.0, 0.0, 1.0],
        "right_end_position": [0.3, -0.2, 0.5],
        "right_end_orientation": [0.0, 0.0, 0.0, 1.0]
      },
      "gripper_position": [0.0, 0.0]
    }
  ],
  "action_type": "pose",
  "action_freq": 30.0
}
```

#### Action dict fields

| Field                                    | Type         | Required | Description                                          |
|------------------------------------------|--------------|----------|------------------------------------------------------|
| robot_position                           | dict         | Yes      | Joint or pose targets (see below)                    |
| robot_position.arm_joint_position        | list[float]  | joint    | (14,) arm motor positions in radians                 |
| robot_position.head_joint_position       | list[float]  | joint    | (3,) head motor positions in radians                 |
| robot_position.waist_joint_position      | list[float]  | joint    | (5,) waist motor positions in radians                |
| robot_position.left_end_position         | list[float]  | pose     | (3,) [x, y, z] in metres                            |
| robot_position.left_end_orientation      | list[float]  | pose     | (4,) [qx, qy, qz, qw] quaternion                   |
| robot_position.right_end_position        | list[float]  | pose     | (3,) [x, y, z] in metres                            |
| robot_position.right_end_orientation     | list[float]  | pose     | (4,) [qx, qy, qz, qw] quaternion                   |
| gripper_position                         | list[float]  | No       | (2,) [left, right] gripper positions                 |
| chassis_velocity                         | list[float]  | No       | (3,) [vx, vy, wz] chassis velocity                  |

#### Response Example

```json
{
  "status": "accepted"
}
```

#### Response Fields

| Field  | Type   | Description                                 |
|--------|--------|---------------------------------------------|
| status | string | `"accepted"` — actions queued for execution |

---

### Stop Motion

**Endpoint:** `/stop_motion`
**Method:** `POST`

Cancel all pending actions on the robot.

#### Request Parameters

None (session is identified automatically).

#### Response Example

```json
{
  "status": "success"
}
```

#### Response Fields

| Field  | Type   | Description                                                  |
|--------|--------|--------------------------------------------------------------|
| status | string | `"success"` — all pending actions cancelled                  |

---

### Get Status

**Endpoint:** `/status`
**Method:** `GET`

Query the current motion playback status.

#### Request Parameters

None (session is identified automatically).

#### Response Example

```json
{
  "action_queue": 120,
  "navigation_queue": 0,
  "resetting": false,
  "status": "pending actions"
}
```

#### Response Fields

| Field            | Type   | Description                                                                  |
|------------------|--------|------------------------------------------------------------------------------|
| action_queue     | int    | Remaining action frames yet to be sent to the robot                          |
| navigation_queue | int    | Navigation targets in flight or queued                                       |
| resetting        | bool   | Whether a reset operation is in progress                                     |
| status           | string | `"free"`, `"resetting"`, `"navigating"`, `"pending actions"`, or `"pending"` |

Status priority: `resetting` > actively `navigating` > actively `pending actions` > `pending` (queued, waiting for lock) > `free`.

---

### Navigate to Position

**Endpoint:** `/goto_navi_position`
**Method:** `POST`

Navigate the robot chassis to a map-frame goal. The server runs the navigation in the background; the response returns immediately.

#### Request Parameters

| Parameter  | Type | Required | Default | Description                                                                |
|------------|------|----------|---------|----------------------------------------------------------------------------|
| position   | dict | Yes      | —       | Goal with keys `position` [x, y, z] m and `orientation` [qx, qy, qz, qw] |
| max_refine | int  | No       | 5       | Max chassis refinement iterations (1–100)                                  |

The HTTP body should be a JSON object:

```json
{
  "position": {
    "position": [1.0, 2.0, 0.0],
    "orientation": [0.0, 0.0, 0.0, 1.0]
  },
  "max_refine": 5
}
```

#### Response Example

```json
{
  "status": "accepted"
}
```

#### Response Fields

| Field  | Type   | Description                                                 |
|--------|--------|-------------------------------------------------------------|
| status | string | `"accepted"` — navigation goal queued for background execution |

---

## Robot Specific Notes

### G2 (Dual-Arm Humanoid)

- Dual-arm humanoid robot with mobile chassis
- 14 DOF arms (7 joints × 2 arms)
- 3 DOF head, 5 DOF waist
- 2 grippers (left, right), range **[-0.91, 0.0]** (0.0 = fully open, -0.91 = fully closed)

**Joint control:**
- `arm_joint_position`: 14 floats — `[left 7 joints, right 7 joints]` in radians
- `head_joint_position`: 3 floats in radians
- `waist_joint_position`: 5 floats in radians
- `gripper_position`: 2 floats — `[left, right]`, range [-0.91, 0.0]

**Pose control:**
- `left_end_position`: `[x, y, z]` in metres (base_link frame)
- `left_end_orientation`: `[qx, qy, qz, qw]` quaternion
- `right_end_position`: `[x, y, z]` in metres (base_link frame)
- `right_end_orientation`: `[qx, qy, qz, qw]` quaternion

**Chassis control (optional in actions):**
- `chassis_velocity`: `[vx, vy, wz]` — linear x, linear y, angular z

**Joint limits (radians):**

`waist_joint_position` (5 joints):

| Index | Joint Name    | Min (rad) | Max (rad) |
|-------|---------------|-----------|-----------|
| 0     | body_joint1   | -1.0821   | 0.0002    |
| 1     | body_joint2   | -0.0002   | 2.6529    |
| 2     | body_joint3   | -1.9199   | 1.0390    |
| 3     | body_joint4   | -0.4363   | 0.4363    |
| 4     | body_joint5   | -3.0456   | 3.0456    |

`head_joint_position` (3 joints):

| Index | Joint Name    | Min (rad) | Max (rad) |
|-------|---------------|-----------|-----------|
| 0     | head_joint1   | -0.7810   | 0.7810    |
| 1     | head_joint2   | -0.1510   | 0.1510    |
| 2     | head_joint3   | -0.0180   | 0.5230    |

`arm_joint_position` (14 joints = left 7 + right 7, left and right share the same limits):

| Index (L/R) | Joint Name  | Min (rad) | Max (rad) |
|-------------|-------------|-----------|-----------|
| 0 / 7       | arm_joint1  | -3.0718   | 3.0718    |
| 1 / 8       | arm_joint2  | -2.0595   | 2.0595    |
| 2 / 9       | arm_joint3  | -3.0718   | 3.0718    |
| 3 / 10      | arm_joint4  | -2.4958   | 1.0123    |
| 4 / 11      | arm_joint5  | -3.0718   | 3.0718    |
| 5 / 12      | arm_joint6  | -1.0123   | 1.0123    |
| 6 / 13      | arm_joint7  | -1.5359   | 1.5359    |

`gripper_position` (2 values):

| Index | Side  | Min   | Max |
|-------|-------|-------|-----|
| 0     | Left  | -0.91 | 0.0 |
| 1     | Right | -0.91 | 0.0 |

> Values outside these limits are automatically clamped by the server before execution.

**Cameras:**

| Name              | Type  | Resolution  | Description          |
|-------------------|-------|-------------|----------------------|
| kHeadColor        | RGB   | 640 × 400   | Head-mounted color   |
| kHeadDepth        | Z16   | 640 × 400   | Head-mounted depth   |
| kHandLeftColor    | RGB   | 1280 × 1056 | Left arm-mounted     |
| kHandRightColor   | RGB   | 1280 × 1056 | Right arm-mounted    |
| kHeadStereoLeft   | RGB   | —           | Head stereo (left)   |
| kHeadStereoRight  | RGB   | —           | Head stereo (right)  |

---

## Contact

For official inquiries or support, you can reach us via:
- **GitHub Issues:** [https://github.com/RoboChallenge/RoboChallengeInference/issues](https://github.com/RoboChallenge/RoboChallengeInference/issues)
- **Reddit:** [https://www.reddit.com/r/RoboChallenge/](https://www.reddit.com/r/RoboChallenge/)
- **Discord:** [https://discord.gg/8pD8QWDv](https://discord.gg/8pD8QWDv)
- **X (Twitter):** [https://x.com/RoboChallengeAI](https://x.com/RoboChallengeAI)
- **HuggingFace:** [https://huggingface.co/RoboChallenge](https://huggingface.co/RoboChallenge)
- **GitHub:** [https://github.com/RoboChallenge](https://github.com/RoboChallenge)
- **Support Email:** [support@robochallenge.ai](mailto:support@robochallenge.ai)
