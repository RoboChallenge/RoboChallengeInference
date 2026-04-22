
# CVPR Pi0.5 Multi-Task Example

This branch is a **CVPR on-robot inference example** for **Pi0.5 multi-task** policies. There is **one checkpoint per robot platform**; you can evaluate **all tasks** supported on that platform with the **same model**, by changing the **task prompt** (and episode data) only. Supported robot types in this codebase (see `demo.py` / `test.py`): `dosw`, `aloha`, `arx5`, `ur5`. 

Core entrypoints:

- **`test.py`** — Connects to the local **mock robot server** for quick debugging.
- **`demo.py`** — **RoboChallenge** official evaluation (requires `user_token` and `submission_id`).


## Quick start

### 1. Installation

```bash
uv venv
source .venv/bin/activate   # if your uv venv uses another path, activate that instead
uv pip install -e ./openpi
uv pip install pytest
uv pip install -r requirements.txt
```

### 2. Download the example model

Download the **Pi0.5 multi-task** checkpoint from [RoboChallenge on Hugging Face](https://huggingface.co/RoboChallenge/models). **One checkpoint per embodiment** covers **all tasks** on that robot; switch tasks via `--prompt` only.

| Platform |`robot_type` (in `test.py` / `demo.py`) | model_name | Mock `ROBOT_TAG` (mock test section) |
|----------|--------------------------------------------|-----------------------------|---------------------------|
| W1 | `dosw` | `table30v2_multitask_baseline_w1` | `w1` |
| Aloha | `aloha` | `table30v2_multitask_baseline_aloha` | `aloha` |
| ARX5 | `arx5` | `table30v2_multitask_baseline_arx5` | `arx5` |
| UR5 | `ur5` | `table30v2_multitask_baseline_ur5` | `ur5` |


### 3. Run the mock test

1. Edit `mock_server/mock_settings.py`: set **`ROBOT_TAG`** and **`RECORD_DATA_DIR`** (only one active pair). With the bundled sample data (paths relative to repo root):

   - `ROBOT_TAG='aloha'`, `RECORD_DATA_DIR='../20260413/aloha/pack_the_toothbrush_holder'`
   - `ROBOT_TAG='w1'`, `RECORD_DATA_DIR='../20260413/w1/sweep_the_trash'` → run **`test.py` with `--robot_type dosw`**
   - `ROBOT_TAG='ur5'`, `RECORD_DATA_DIR='../20260413/ur5/arrange_fruits'`
   - `ROBOT_TAG='arx5'`, `RECORD_DATA_DIR='../20260413/arx5/hang_the_cup'`

2. Terminal 1 — start the mock server:

    ```bash
    cd mock_server
    python3 mock_robot_server.py
    ```

3. Terminal 2 — from the repo root:

    ```bash
    python3 test.py \
      --checkpoint /path/to/checkpoint_dir \
      --prompt "task instruction in natural language(match the format used in training)" \
      --robot_type dosw
    ```

    Use exactly one of `--robot_type dosw | aloha | arx5 | ur5` so it matches `ROBOT_TAG`.

    Adjust `--action_type`, `--duration`, `--image_size`, etc. as needed (see `test.py`).

### 4. Run the demo

After you have a submission and are in the assigned evaluation window, run **`demo.py`**.


**W1**

```bash
python3 demo.py \
  --user_token <your_user_token> \
  --submission_id <your_submission_id> \
  --checkpoint /path/to/w1_checkpoint_dir \
  --prompt "task instruction in natural language(match the format used in training)" \
  --action_type joint \
  --image_size "640x480" \
  --robot_type dosw
```

**Aloha**

```bash
python3 demo.py \
  --user_token <your_user_token> \
  --submission_id <your_submission_id> \
  --checkpoint /path/to/aloha_checkpoint_dir \
  --prompt "task instruction in natural language(match the format used in training)" \
  --action_type joint \
  --image_size "640x480" \
  --robot_type aloha
```

**ARX5**

```bash
python3 demo.py \
  --user_token <your_user_token> \
  --submission_id <your_submission_id> \
  --checkpoint /path/to/arx5_checkpoint_dir \
  --prompt "task instruction in natural language(match the format used in training)" \
  --action_type leftjoint \
  --image_size "1280x720" \
  --robot_type arx5
```

**UR5**

```bash
python3 demo.py \
  --user_token <your_user_token> \
  --submission_id <your_submission_id> \
  --checkpoint /path/to/ur5_checkpoint_dir \
  --prompt "task instruction in natural language(match the format used in training)" \
  --action_type leftjoint \
  --image_size "640x480" \
  --robot_type ur5
```


# Original README.md

# RoboChallengeInference

## Project Structure

```
- RoboChallengeInference/
    - README.md
    - requirements.txt
    - demo.py
    - test.py  # Main test entry script
    - robot/
        - __init__.py
        - interface_client.py
        - job_worker.py
    - mock_server
        - mock_rc_robot.py
        - mock_robot_server.py
        - mock_settings.py
        - utils.py
    - utils/
        - __init__.py
        - enums.py
        - log.py
        - util.py
```

## User Guide

### 1. Installation

```bash
# Clone the repository and checkout the specified branch
git clone https://github.com/RoboChallenge/RoboChallengeInference.git
cd RoboChallengeInference

# (Recommended) Create and activate a virtual environment to avoid polluting your global Python environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

```

### 2. Checkout & Modification

```bash
# Checkout
git checkout -b my-feature-branch
# Follow the instructions in demo.py to modify parameters and implement your custom inference logic based on DummyPolicy.
# The current task prompt will be passed into `DummyPolicy.run_policy(input_data, prompt=...)`.
```

### 3. Test

```bash
# Open the mock_settings.py file and set the ROBOT_TAG and RECORD_DATA_DIR variables according to your robot and data directory requirements.
# Notes:
#   Only one pair of ROBOT_TAG and RECORD_DATA_DIR should be active at a time.
#   Ensure that the RECORD_DATA_DIR path matches the structure of your data folder.
#   You can find the appropriate ROBOT_TAG in your training data or on our website.
#   For the 20260413 CVPR package, you can use one of the following pairs:
#     ROBOT_TAG='aloha', RECORD_DATA_DIR='../20260413/aloha/pack_the_toothbrush_holder'
#     ROBOT_TAG='w1',    RECORD_DATA_DIR='../20260413/w1/sweep_the_trash'
#     ROBOT_TAG='ur5',   RECORD_DATA_DIR='../20260413/ur5/arrange_fruits'
#     ROBOT_TAG='arx5',  RECORD_DATA_DIR='../20260413/arx5/hang_the_cup'
#   RECORD_DATA_DIR also supports robot-level directory (e.g. '../20260413/ur5').
#   The mock server will auto-detect the task directory with meta/states/videos.
# Start the test service
cd mock_server
python3 mock_robot_server.py
# Use test.py for testing; it will automatically invoke the mock interface to help you debug your model
# Replace {your_args} with the actual parameters you want to test, for example: --checkpoint xxx.
# Run this in another shell at repo root.
python3 test.py {your_args}
```

### 4. Submit

- Log in to RoboChallenge Web
- Submit an evaluation request
- On the "My Submission" page, you can view your submissions. Click "Detail" to see more information about a submission.
- The Submission ID displayed on the details page will be required for the evaluation process. The program will automatically poll and select active runs under that submission.

### 5. Execute

- Wait for a notification (on the website or via email) indicating that your task has been assigned.
- Ensure the modified code from the previous steps is actively running during the assigned period.
- Start the evaluation worker with:

```bash
python3 demo.py --user_token <your_user_token> --submission_id <your_submission_id> --checkpoint <your_checkpoint>
```

- After the task is completed, the program will exit normally. If you encounter any issues or exceptions, please feel
  free to contact us.

### 6. Result

Once your task has been executed, you can view the results by visiting the "My Submissions" page on the website.

## Key API Parameter Descriptions

This is the direct interface for the robot.
The base URL is `/api/robot/<id>/direct`. For example, if the robot ID is `1`, the full URL to get the state is
`/api/robot/1/direct/state.pkl`.

### Sync Clock

**Endpoint:** `/clock-sync`  
**Method:** `GET`

#### Request Parameters

None

#### Response Example

```json
{
  "timestamp": 0.0
}
```

#### Response Fields

| Field     | Type  | Description                 |
|-----------|-------|-----------------------------|
| timestamp | float | unix timestamp on the robot |

---

### Get State

**Endpoint:** `/state.pkl`  
**Method:** `GET`

#### Request Parameters

| Parameter   | Type        | Required | Default | Description                                                                                                                                                                                                                                                                                                                                                                                            |
|-------------|-------------|----------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| width       | integer     | No       | 224     | Width of the image                                                                                                                                                                                                                                                                                                                                                                                     |
| height      | integer     | No       | 224     | Height of the image                                                                                                                                                                                                                                                                                                                                                                                    |
| image_type  | list of str | Yes      | None    | Camera names. Only robot-specific `cam_*` keys listed below are supported. If you send unsupported keys, server returns `JSONResponse` error with valid options.                                                                                                                                                                                                                                    |
| action_type | str         | Yes      | None    | Control mode. Only robot-specific values listed below are supported. If you send unsupported values, server returns `JSONResponse` error with valid options.                                                                                                                                                                                                                                          |

Robot-specific `image_type` values and returned `images` keys:

| Robot | Use this `image_type` | `images` keys returned |
|-------|------------------------|------------------------|
| `aloha` | `cam_left_wrist`, `cam_right_wrist`, `cam_high` | `cam_left_wrist`, `cam_right_wrist`, `cam_high` |
| `w1` | `cam_left_wrist`, `cam_right_wrist`, `cam_high` | `cam_left_wrist`, `cam_right_wrist`, `cam_high` |
| `ur5` | `cam_global`, `cam_arm` | `cam_global`, `cam_arm` |
| `arx5` | `cam_global`, `cam_arm`, `cam_side` | `cam_global`, `cam_arm`, `cam_side` |

Robot-specific `action_type` values:

| Robot | Supported `action_type` |
|-------|--------------------------|
| `aloha` | `joint`, `pos`, `leftjoint`, `leftpos`, `rightjoint`, `rightpos` |
| `w1` | `joint`, `pos`, `leftjoint`, `leftpos`, `rightjoint`, `rightpos` |
| `ur5` | `leftjoint`, `leftpos` |
| `arx5` | `leftjoint`, `leftpos` |

#### Response Example

The response is a pickle file containing a dictionary with the following structure.
The `images` keys depend on robot type:

- `aloha` / `w1`

```python
{
    "images": {
        "cam_left_wrist": b'PNG',
        "cam_right_wrist": b'PNG',
        "cam_high": b'PNG'
    }
}
```

- `ur5`

```python
{
    "images": {
        "cam_global": b'PNG',
        "cam_arm": b'PNG'
    }
}
```

- `arx` (`arx5`)

```python
{
    "images": {
        "cam_global": b'PNG',
        "cam_arm": b'PNG',
        "cam_side": b'PNG'
    }
}
```

Full response example (`aloha`, `action_type=joint`):

```python
{
    "state": 'normal',
    "timestamp": 1774949968.382069,
    "pending_actions": 0,
    "action": [
        -0.0151, 0.0, 0.0, -0.0184, 0.0986, 0.0529, 0.0001,
         0.0113, 0.0, 0.0, -0.0079, 0.0962, 0.0298, 0.0
    ],
    "images": {
        "cam_left_wrist": b'PNG',
        "cam_right_wrist": b'PNG',
        "cam_high": b'PNG'
    }
}
```

#### Response Fields

| Field              | Type          | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|--------------------|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| state              | string        | Robot state. Should be `normal` if the robot is operational. If the value is `fault` or `abnormal`, there is an issue with the robot. If the value is `size_none`, the request parameter `image_type` or `action_type` is missing.                                                                                                                                                                                                                                                                                                                                                                                |
| timestamp          | float         | Unix timestamp on the robot                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| pending_actions    | integer       | Number of pending actions in the queue                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| action             | list of float | Current robot joint or position values. If `action_type` in the request contains `joint`, the joint values will be returned. If it contains `pos`, the tool end positions will be returned. If it contains `left` or `right`, only the values for the left or right arm will be returned. If neither is specified, values for both arms will be returned. For example, if the robot is Aloha with two arm, the list consists with `[joints of left arm, gripper of left arm, joints of right arm, gripper of right arm]`. See the [Robot specific Notes](#robot-specific-notes) section for detailed information. |
| images             | dict          | Dictionary of images. Only includes camera positions specified in the `image_type` request parameter.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| images.cam_left_wrist  | bytes         | PNG image bytes for `aloha`/`w1`, if requested                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| images.cam_right_wrist | bytes         | PNG image bytes for `aloha`/`w1`, if requested                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| images.cam_high        | bytes         | PNG image bytes for `aloha`/`w1`, if requested                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| images.cam_global      | bytes         | PNG image bytes for `ur5`/`arx`, if requested                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| images.cam_arm         | bytes         | PNG image bytes for `ur5`/`arx`, if requested                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| images.cam_side        | bytes         | PNG image bytes for `arx`, if requested                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |

---

### Post Action

**Endpoint:** `/action`  
**Method:** `POST`

#### Request Parameters

| Parameter   | Type | Required | Default | Description                                                                                                                                                                                                                                       |
|-------------|------|----------|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| action_type | str  | Yes      | None    | Control mode. Only robot-specific values listed above are supported. If you send unsupported values, server returns `JSONResponse` error with valid options. |

The HTTP body should be a JSON object with the following structure:

```json
{
  "actions": [
    [
      0.0,
      0.0
    ],
    [
      0.0,
      0.0
    ],
    [
      0.0,
      0.0
    ]
  ],
  "duration": 0.0
}
```

| Field    | Type          | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|----------|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| actions  | 2D float list | Target joint or position values. If `action_type` in the request contains `joint`, the target values control the robot joints. If it contains `pos`, the tool end positions will be controlled. If it contains `left` or `right`, only the left or right arm will be controlled. If neither is specified, both arms will be controlled. The shape of the array is (number of actions, target values per action). For example, if you are using ALOHA and `action_type` is `joint`, then the shape of the actions array should be (N, 14): 6 joints and 1 gripper per arm, N is the number of steps your model infers. See the [Robot specific Notes](#robot-specific-notes) section for detailed information. |
| duration | float         | Duration (second) per action                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

#### Response Example

```json
{
  "result": "success",
  "message": ""
}

```

#### Response Fields

| Field   | Type   | Description                                                                                                                                                        |
|---------|--------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| result  | string | Result of the request. Only `success` or `error` will be returned.                                                                                                 |
| message | string | Reason for `error` result, if any. possible message: the robot is not running (fault or logging), the action shape is wrong, action queue is full, other exception |

## Robot specific Notes

Different robots have different action shapes and camera placement.

- W1
    - Dual-arm robot
    - 7 DOF per arm (6 joints + 1 gripper)
    - Joint control:
        - one arm(left or right): 7 numbers total: `[6 joints, 1 gripper]`
        - two arms: 14 numbers total: `[left 6 joints, left 1 gripper, right 6 joints, right 1 gripper]`
    - Pose control
        - one arm(left or right): 8 numbers total: `[x, y, z, quaternion(xyzw), gripper]`
        - two arms: 16 numbers
          total:
          `[left x, left y, left z, left quaternion(xyzw), left gripper, right x, right y, right z, right quaternion(xyzw), right gripper]`
    - 3 cameras: mounted on left/right arm, and on the top of the robot

- Aloha
    - Dual-arm robot
        - 7 DOF per arm (6 joints + 1 gripper)
        - Joint control:
            - one arm(left or right): 7 numbers total: `[6 joints, 1 gripper]`
            - two arms: 14 numbers total: `[left 6 joints, left 1 gripper, right 6 joints, right 1 gripper]`
        - Pose control
            - one arm(left or right): 8 numbers total: `[x, y, z, quaternion(xyzw), gripper]`
            - two arms: 16 numbers
              total:
              `[left x, left y, left z, left quaternion(xyzw), left gripper, right x, right y, right z, right quaternion(xyzw), right gripper]`
        - 3 cameras: mounted on left/right arm, and on the top of the robot


- Arx5
    - Single-arm robot
    - 7 DOF (6 joints + 1 gripper)
    - Joint control: 7 numbers total: `[6 joints, 1 gripper]`
    - Pose control: 8 numbers total: `[x, y, z, quaternion(xyzw), gripper]`
    - You must always use `left` in the `action_type` parameter, e.g., `leftjoint` or `leftpos`.
    - 3 cameras: mounted on the arm, opposite to the arm, and on the right side of the arm

- Ur5
    - Single-arm robot
    - 7 DOF (6 joints + 1 gripper)
    - Joint control: 7 numbers total: `[6 joints, 1 gripper]`
    - Pose control: 8 numbers total: `[x, y, z, quaternion(xyzw), gripper]`
    - You must always use `left` in the `action_type` parameter, e.g., `leftjoint` or `leftpos`.
    - 2 cameras: mounted on the arm, and opposite to the arm

## Contact

For official inquiries or support, you can reach us via:
- **GitHub Issues:** [https://github.com/RoboChallenge/RoboChallengeInference/issues](https://github.com/RoboChallenge/RoboChallengeInference/issues)
- **Reddit:** [https://www.reddit.com/r/RoboChallenge/](https://www.reddit.com/r/RoboChallenge/)
- **Discord:** [https://discord.gg/8pD8QWDv](https://discord.gg/8pD8QWDv)
- **X (Twitter):** [https://x.com/RoboChallengeAI](https://x.com/RoboChallengeAI)
- **HuggingFace:** [https://huggingface.co/RoboChallenge](https://huggingface.co/RoboChallenge)
- **GitHub:** [https://github.com/RoboChallenge](https://github.com/RoboChallenge)
- **Support Email:** [support@robochallenge.ai](mailto:support@robochallenge.ai)
