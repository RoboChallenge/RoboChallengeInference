import argparse
import logging
import cv2
import numpy as np

from robot.interface_client import InterfaceClient
from robot.job_worker import job_loop

from openpi_rtc.policies import policy_config as _policy_config
from openpi_rtc.training import config as _config
from openpi_rtc import transforms as _transforms


logging.basicConfig(
    filename='mylogfile.log',  # Log file name
    level=logging.INFO,  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s %(levelname)s:%(message)s'  # Log format
)


class OpenpiPolicy:
    """
    Example policy class.
    Users should implement the __init__ and run_policy methods according to their own logic.
    """

    def __init__(
        self,
        config_name,
        checkpoint_dir,
        prompt,
        adarms_knob=0,
        valid_action_num=16,
        duration=0.05,
        action_type="joint",
        image_size=(224, 224),
        robot_type="arx5",
    ):
        config = _config.get_config(config_name)
        self.policy = _policy_config.create_trained_policy(config, checkpoint_dir, repack_transforms=_transforms.Group(),)
        self.prompt = prompt
        self.adarms_knob = np.int32(adarms_knob)
        self.valid_action_num = valid_action_num
        self.duration = duration
        self.action_type = action_type
        self.robot_type = robot_type
        self.delta_mask = np.asarray(_transforms.make_bool_mask(6, -1, 6, -1), dtype=bool)
        self.output_has_absolute = self._policy_has_absolute_actions()
        image_mapping = {
            "aloha": {
                "cam_high": "observation/cam_high",
                "cam_left_wrist": "observation/cam_wrist_left",
                "cam_right_wrist": "observation/cam_wrist_right",
            },
            "dosw": {
                "cam_high": "observation/cam_high",
                "cam_left_wrist": "observation/cam_wrist_left",
                "cam_right_wrist": "observation/cam_wrist_right",
            },
            "arx5": {
                "cam_global": "observation/cam_high",
                "cam_arm": "observation/cam_wrist_left",
                "cam_side": "observation/cam_wrist_right",
            },
            "ur5": {
                "cam_global": "observation/cam_high",
                "cam_arm": "observation/cam_wrist_left",
            },
        }
        self.key_mapping = image_mapping[self.robot_type]
        self.inp_images = {key: np.zeros((image_size[1], image_size[0], 3)) for key in self.key_mapping.values()}
        
        print(f"action type: {self.action_type}, robot type: {self.robot_type}")
        if self.robot_type == "aloha" or self.robot_type == "dosw":
            inp_state = np.zeros(14, dtype=np.float32)
        else:
            inp_state = np.zeros(7, dtype=np.float32)

        # warm up model
        self.policy.infer(
            {
                **self.inp_images,
                "state": inp_state,
                "prompt": self.prompt,
                "adarms_knob": self.adarms_knob
            }
        )

    def _policy_has_absolute_actions(self) -> bool:
        output_transform = getattr(self.policy, "_output_transform", None)
        transforms_list = getattr(output_transform, "transforms", None)
        if transforms_list is None:
            return False
        return any(isinstance(t, _transforms.AbsoluteActions) for t in transforms_list)

    def _to_absolute_actions(self, actions: np.ndarray, obs_state: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        obs_state = np.asarray(obs_state, dtype=np.float32)
        dims = min(actions.shape[-1], obs_state.shape[-1], self.delta_mask.shape[0])
        if dims <= 0:
            return actions
        actions = actions.copy()
        state_offset = obs_state[:dims] * self.delta_mask[:dims]
        actions[..., :dims] += state_offset
        return actions

    def process_images(self, state):
        for source in self.key_mapping.keys():
            if source in state["images"]:
                image_data = state["images"][source]
                image = cv2.imdecode(
                    np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED
                )
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                final_key = self.key_mapping.get(source, source)
                self.inp_images[final_key] = image

    def process_actions_for_model(self, actions):
        actions_array = np.array(actions, dtype=np.float32)
        if self.robot_type == "ur5":
            actions_array[6] = 3 + (0.085 - actions_array[6]) * (226 / 0.085)
        return actions_array

    def process_actions_for_robot(self, actions):
        actions_array = np.array(actions)
        if self.robot_type == "dosw":
            actions_array[:, 6] -= 0.002
            actions_array[:, 13] -= 0.002
        elif self.robot_type == "ur5":
            actions_array[:, 6] = 0.085 - (actions_array[:, 6] - 3) * 0.085 / 226
        return actions_array

    def run_policy(self, state, prompt=None):
        # processed images are stored in self.inp_images
        self.process_images(state)
        obs_state = self.process_actions_for_model(state["action"])
        inputs = {
            **self.inp_images,
            "state": obs_state,
            "prompt": self.prompt if prompt is None else prompt,
            "adarms_knob": self.adarms_knob
        }
        print(f"input state: {inputs['state']}")
        result = self.policy.infer(inputs)["actions"]
        if not self.output_has_absolute:
            result = self._to_absolute_actions(result, obs_state)
        if self.valid_action_num is not None:
            result = result[0:self.valid_action_num]
        actions_for_robot = self.process_actions_for_robot(result)
        logging.info(f"gripper_actions_for_robot: {actions_for_robot[:,-2:].tolist()}")
        return actions_for_robot.tolist()


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

    def infer(self, state, prompt=None):
        """
        Main entry point for inference.
        Args:
            state: Input state for the policy. Refer to README.md#get-state response example for details. It's unpickled and passed as a dict here.
            prompt (str, optional): Task prompt for the current run.
        Returns:
            list: Inference results from the policy. Refer to README.md#post-action request parameters for details. This will be the `actions` field in the request.
        """
        result = self.policy.run_policy(state, prompt=prompt)
        return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--user_token', type=str, required=True, help='User token')
    parser.add_argument('--submission_id', type=str, required=True, help='Submission ID. Get it from the detail page of your submission')
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

    # these args are generally not changed during evaluation, so we put them here.
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
    

    client = InterfaceClient(args.user_token)

    if args.robot_type == "dosw":
        config_name = "cvpr_multitask_dosw1_rtc"
    elif args.robot_type == "aloha":
        config_name = "cvpr_multitask_aloha_rtc"
    elif args.robot_type == "arx5":
        config_name = "cvpr_multitask_arx5_rtc"
    elif args.robot_type == "ur5":
        config_name = "cvpr_multitask_ur5_rtc"
    else:
        raise ValueError(f"Invalid robot type: {args.robot_type}")
    policy = OpenpiPolicy(config_name, args.checkpoint, args.prompt, valid_action_num=valid_action_num, duration=duration, action_type=action_type, image_size=image_size, robot_type=robot_type)
    
    gpu_client = GPUClient(policy)  # add your own parameters

    # main job loop. This function monitors when jobs are ready to eval and do the evaluation
    job_loop(client, gpu_client, args.submission_id, image_size, image_type, action_type, duration)


if __name__ == '__main__':
    main()
