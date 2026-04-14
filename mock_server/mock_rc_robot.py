# -*- coding: utf-8 -*-
import json
import os
import time
from abc import ABC, abstractmethod
from enum import Enum
from threading import Thread
from typing import Any, ClassVar

import cv2


class RobotTag(Enum):
    ALOHA = 'aloha'
    UR5 = 'ur5'
    ARX5 = 'arx5'
    W1 = 'w1'


class MockRCRobot(ABC):
    robot_alpha: Any
    filler_thread: Thread
    frame_interval = 1 / 30
    ROBOT_TYPE: ClassVar[str | None] = None
    IMAGE_TYPES: ClassVar[tuple[str, ...]] = ()
    ACTION_TYPES: ClassVar[tuple[str, ...]] = ()

    @property
    @abstractmethod
    def dof_num(self):
        raise NotImplemented('no joint number defined')

    @property
    @abstractmethod
    def pos_num(self):
        raise NotImplemented('no pose number defined')

    @classmethod
    def _iter_subclasses(cls):
        for subclass in cls.__subclasses__():
            yield subclass
            yield from subclass._iter_subclasses()

    @classmethod
    def available_robot_types(cls) -> list[str]:
        robot_types = []
        for subclass in cls._iter_subclasses():
            robot_type = getattr(subclass, "ROBOT_TYPE", None)
            if robot_type:
                robot_types.append(robot_type)
        return sorted(set(robot_types))

    @property
    def robot_type(self) -> str:
        if not self.ROBOT_TYPE:
            raise RuntimeError(f'{type(self).__name__} has no ROBOT_TYPE')
        return self.ROBOT_TYPE

    @property
    def image_types(self) -> list[str]:
        if not self.IMAGE_TYPES:
            raise RuntimeError(f'{type(self).__name__} has no IMAGE_TYPES')
        return list(self.IMAGE_TYPES)

    @property
    def action_types(self) -> list[str]:
        if not self.ACTION_TYPES:
            raise RuntimeError(f'{type(self).__name__} has no ACTION_TYPES')
        return list(self.ACTION_TYPES)

    @property
    def image_index(self) -> dict[str, int]:
        return {image_name: idx for idx, image_name in enumerate(self.image_types)}

    def schema(self) -> dict[str, Any]:
        robot_type = self.robot_type
        image_types = self.image_types
        action_types = self.action_types
        image_index = self.image_index

        if len(image_types) != len(set(image_types)):
            raise ValueError(f'{type(self).__name__} has duplicated IMAGE_TYPES: {image_types}')
        if not action_types:
            raise ValueError(f'{type(self).__name__} has empty ACTION_TYPES')
        expected_index = list(range(len(image_types)))
        if sorted(image_index.values()) != expected_index:
            raise ValueError(
                f'{type(self).__name__} image_index is invalid: {image_index}, expected indices: {expected_index}'
            )
        if set(image_index.keys()) != set(image_types):
            raise ValueError(
                f'{type(self).__name__} image_index keys mismatch, image_types={image_types}, image_index={image_index}'
            )

        return {
            "robot_type": robot_type,
            "available_image_type": image_types,
            "available_action_type": action_types,
            "image_index": image_index,
        }

    @staticmethod
    def create_robot(robot_tag: RobotTag | str, realsense_ids, record_data_dir) -> 'MockRCRobot':
        robot_tag = RobotTag(robot_tag)
        if robot_tag == RobotTag.ALOHA:
            return MockRCRobotAloha(robot_tag, realsense_ids, record_data_dir)
        if robot_tag == RobotTag.W1:
            return MockRCRobotW1(robot_tag, realsense_ids, record_data_dir)
        if robot_tag == RobotTag.ARX5:
            return MockRCRobotArx5(robot_tag, realsense_ids, record_data_dir)
        if robot_tag == RobotTag.UR5:
            return MockRCRobotUr5(robot_tag, realsense_ids, record_data_dir)
        raise RuntimeError('unknown robot tag')

    @staticmethod
    def iter_jsonl(file_path):
        while True:
            with open(file_path, 'r') as file_obj:
                for line in file_obj:
                    yield json.loads(line)

    @staticmethod
    def iter_mp4(file_path):
        while True:
            cap = cv2.VideoCapture(file_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
            cap.release()

    @abstractmethod
    def fill(self):
        pass

    def filler(self):
        for _ in range(max(self.get_frame_number() - 1, 0)):
            start_time = time.time()
            self.fill()
            try:
                time.sleep(self.frame_interval - (time.time() - start_time))
            except ValueError:
                pass

    def get_frame_number(self):
        with open(f'{self.record_data_dir}/meta/episode_meta.json', 'r') as file_obj:
            meta = json.load(file_obj)
            return meta['frames']

    @staticmethod
    def _resolve_record_data_dir(record_data_dir: str) -> str:
        record_data_dir = os.path.expanduser(record_data_dir)
        if os.path.isdir(record_data_dir):
            if all(os.path.isdir(os.path.join(record_data_dir, item)) for item in ('meta', 'states', 'videos')):
                return record_data_dir
            for sub_name in sorted(os.listdir(record_data_dir)):
                sub_path = os.path.join(record_data_dir, sub_name)
                if not os.path.isdir(sub_path):
                    continue
                if all(os.path.isdir(os.path.join(sub_path, item)) for item in ('meta', 'states', 'videos')):
                    return sub_path
        raise FileNotFoundError(f'Cannot resolve record data dir from: {record_data_dir}')

    def _find_data_file(self, relative_path: str) -> str:
        abs_path = os.path.join(self.record_data_dir, relative_path)
        if os.path.exists(abs_path):
            return abs_path
        raise FileNotFoundError(f'Cannot find file under {self.record_data_dir}: {relative_path}')

    def __init__(self, robot_tag: RobotTag | str, realsense_ids, record_data_dir):
        self.robot_tag = RobotTag(robot_tag)
        self.record_data_dir = self._resolve_record_data_dir(record_data_dir)
        self._record_started = False

    def left_get_enable(self):
        return True

    def right_get_enable(self):
        return True

    def left_get_joint(self):
        pass

    def right_get_joint(self):
        pass

    def left_get_pose(self):
        pass

    def right_get_pose(self):
        pass

    def left_go_joint(self, action):
        pass

    def right_go_joint(self, action):
        pass

    def left_go_pose(self, action):
        pass

    def right_go_pose(self, action):
        pass

    def get_imgs(self):
        pass

    def _start_record(self):
        if self._record_started:
            return
        self.filler_thread.start()
        self._record_started = True

    def _stop_record(self):
        pass

    def go_home(self):
        pass

    def go_reset(self):
        pass

    def terminate(self):
        pass


class MockRCRobotAloha(MockRCRobot):
    ROBOT_TYPE = "aloha"
    IMAGE_TYPES = ("cam_left_wrist", "cam_right_wrist", "cam_high")
    ACTION_TYPES = ("joint", "pos", "leftjoint", "leftpos", "rightjoint", "rightpos")

    @property
    def dof_num(self):
        return 6

    @property
    def pos_num(self):
        return 7

    def __init__(self, robot_tag: RobotTag | str, realsense_ids, record_data_dir):
        super().__init__(robot_tag, realsense_ids, record_data_dir)
        self.left_arm_states = self.iter_jsonl(self._find_data_file('states/left_states.jsonl'))
        self.right_arm_states = self.iter_jsonl(self._find_data_file('states/right_states.jsonl'))
        self.left_images = self.iter_mp4(self._find_data_file('videos/cam_left_wrist_rgb.mp4'))
        self.right_images = self.iter_mp4(self._find_data_file('videos/cam_right_wrist_rgb.mp4'))
        self.high_images = self.iter_mp4(self._find_data_file('videos/cam_high_rgb.mp4'))

        self.left_joint = [0.0] * self.dof_num
        self.right_joint = [0.0] * self.dof_num
        self.left_gripper = 0.0
        self.right_gripper = 0.0
        self.left_pose = [0.0] * self.pos_num
        self.right_pose = [0.0] * self.pos_num
        self.frame_left = None
        self.frame_right = None
        self.frame_high = None
        self.filler_thread = Thread(target=self.filler, daemon=True)
        self.fill()

    def fill(self):
        left_state = next(self.left_arm_states)
        right_state = next(self.right_arm_states)
        self.left_joint = left_state['joint_positions']
        self.right_joint = right_state['joint_positions']
        self.left_gripper = float(left_state['gripper_width'])
        self.right_gripper = float(right_state['gripper_width'])
        self.left_pose = [float(item) for item in left_state['ee_positions']]
        self.right_pose = [float(item) for item in right_state['ee_positions']]
        self.frame_left = next(self.left_images)
        self.frame_right = next(self.right_images)
        self.frame_high = next(self.high_images)

    def left_get_joint(self):
        return self.left_joint + [self.left_gripper]

    def right_get_joint(self):
        return self.right_joint + [self.right_gripper]

    def left_get_pose(self):
        return self.left_pose + [self.left_gripper]

    def right_get_pose(self):
        return self.right_pose + [self.right_gripper]

    def get_imgs(self):
        return [
            (time.time(), self.frame_left, None, None),
            (time.time(), self.frame_right, None, None),
            (time.time(), self.frame_high, None, None),
        ]

    def terminate(self):
        pass


class MockRCRobotW1(MockRCRobotAloha):
    ROBOT_TYPE = "w1"
    IMAGE_TYPES = ("cam_left_wrist", "cam_right_wrist", "cam_high")
    ACTION_TYPES = ("joint", "pos", "leftjoint", "leftpos", "rightjoint", "rightpos")


class MockRCRobotArx5(MockRCRobot):
    ROBOT_TYPE = "arx5"
    IMAGE_TYPES = ("cam_global", "cam_arm", "cam_side")
    ACTION_TYPES = ("leftjoint", "leftpos")

    @property
    def dof_num(self):
        return 6

    @property
    def pos_num(self):
        return 7

    def __init__(self, robot_tag: RobotTag | str, realsense_ids, record_data_dir):
        super().__init__(robot_tag, realsense_ids, record_data_dir)
        self.states = self.iter_jsonl(self._find_data_file('states/states.jsonl'))
        self.left_images = self.iter_mp4(self._find_data_file('videos/cam_global_rgb.mp4'))
        self.right_images = self.iter_mp4(self._find_data_file('videos/cam_arm_rgb.mp4'))
        self.high_images = self.iter_mp4(self._find_data_file('videos/cam_side_rgb.mp4'))

        self.joint = [0.0] * self.dof_num
        self.gripper = 0.0
        self.pose = [0.0] * self.pos_num
        self.frame_left = None
        self.frame_right = None
        self.frame_high = None
        self.filler_thread = Thread(target=self.filler, daemon=True)
        self.fill()

    def fill(self):
        state = next(self.states)
        self.joint = state['joint_positions']
        self.gripper = float(state['gripper_width'])
        self.pose = [float(item) for item in state['ee_positions']]
        self.frame_left = next(self.left_images)
        self.frame_right = next(self.right_images)
        self.frame_high = next(self.high_images)

    def left_get_joint(self):
        return self.joint + [self.gripper]

    def right_get_joint(self):
        return self.joint + [self.gripper]

    def left_get_pose(self):
        return self.pose + [self.gripper]

    def right_get_pose(self):
        return self.pose + [self.gripper]

    def get_imgs(self):
        return [
            (time.time(), self.frame_left, None, None),
            (time.time(), self.frame_right, None, None),
            (time.time(), self.frame_high, None, None),
        ]


class MockRCRobotUr5(MockRCRobot):
    ROBOT_TYPE = "ur5"
    IMAGE_TYPES = ("cam_global", "cam_arm")
    ACTION_TYPES = ("leftjoint", "leftpos")

    @property
    def dof_num(self):
        return 6

    @property
    def pos_num(self):
        return 7

    def __init__(self, robot_tag: RobotTag | str, realsense_ids, record_data_dir):
        super().__init__(robot_tag, realsense_ids, record_data_dir)
        self.states = self.iter_jsonl(self._find_data_file('states/states.jsonl'))
        self.left_images = self.iter_mp4(self._find_data_file('videos/cam_global_rgb.mp4'))
        self.right_images = self.iter_mp4(self._find_data_file('videos/cam_arm_rgb.mp4'))

        self.joint = [0.0] * self.dof_num
        self.gripper = 0.0
        self.pose = [0.0] * self.pos_num
        self.frame_left = None
        self.frame_right = None
        self.frame_high = None
        self.filler_thread = Thread(target=self.filler, daemon=True)
        self.fill()

    def fill(self):
        state = next(self.states)
        self.joint = state['joint_positions']
        self.gripper = float(state['gripper_width'])
        self.pose = [float(item) for item in state['ee_positions']]
        self.frame_left = next(self.left_images)
        self.frame_right = next(self.right_images)

    def left_get_joint(self):
        return self.joint + [self.gripper]

    def right_get_joint(self):
        return self.joint + [self.gripper]

    def left_get_pose(self):
        return self.pose + [self.gripper]

    def right_get_pose(self):
        return self.pose + [self.gripper]

    def get_imgs(self):
        return [
            (time.time(), self.frame_left, None, None),
            (time.time(), self.frame_right, None, None),
        ]


if __name__ == '__main__':
    record_dir = '20260413/aloha/pack_the_toothbrush_holder'
    robot = MockRCRobotAloha('aloha', ('233',), record_dir)
    pass
