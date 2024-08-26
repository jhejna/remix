import time
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import tensorflow as tf
from pyquaternion import Quaternion
from widowx_envs.widowx_env_service import WidowXClient

from openx.data.utils import StateEncoding


def state_to_eep(xyz_coor, zangle: float):
    """
    Implement the state to eep function.
    Refered to `bridge_data_robot`'s `widowx_controller/widowx_controller.py`
    return a 4x4 matrix
    """
    assert len(xyz_coor) == 3
    DEFAULT_ROTATION = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
    new_pose = np.eye(4)
    new_pose[:3, -1] = xyz_coor
    new_quat = Quaternion(axis=np.array([0.0, 0.0, 1.0]), angle=zangle) * Quaternion(matrix=DEFAULT_ROTATION)
    new_pose[:3, :3] = new_quat.rotation_matrix
    # yaw, pitch, roll = quat.yaw_pitch_roll
    return new_pose


def wait_for_obs(widowx_client):
    obs = widowx_client.get_observation()
    while obs is None:
        print("Waiting for observations...")
        obs = widowx_client.get_observation()
        time.sleep(1)
    return obs


def convert_obs(obs, resize_shape: Optional[Tuple[int, int]] = None):
    image = obs["full_image"]
    if resize_shape is not None:
        # Make sure we perform resizing exactly as done in the bridge dataset
        image = tf.image.resize(image, resize_shape, method="lanczos3", antialias=True)
        image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
        image = image._numpy()
    return dict(
        image=dict(agent=image),
        state={
            StateEncoding.EE_POS: obs["state"][:3],
            StateEncoding.EE_EULER: obs["state"][3:6],
            StateEncoding.GRIPPER: np.clip(1 - obs["state"][-1:], a_min=0, a_max=1),  # invert gripper
        },
    )


def null_obs(resize_shape: Optional[Tuple[int, int]] = None):
    image_shape = (*resize_shape, 3) if resize_shape is not None else (480, 640, 3)
    return dict(
        image=dict(agent=np.zeros(image_shape)),
        state={
            StateEncoding.EE_POS: np.zeros((3,), dtype=np.float32),
            StateEncoding.EE_EULER: np.zeros((3,), dtype=np.float32),
            StateEncoding.GRIPPER: np.zeros((1,), dtype=np.float32),
        },
    )


def convert_act(act):
    """
    convert the action dict used in openX back to action used in original bridge data robot env
    return: act, shape(7,)
    """
    ee_key = next(k for k in ("achieved_delta", "desired_delta") if k in act)
    bridge_act = np.concatenate(
        [
            act[ee_key][StateEncoding.EE_POS],
            act[ee_key][StateEncoding.EE_EULER],
            np.clip(1 - act["desired_absolute"][StateEncoding.GRIPPER], a_min=0, a_max=1),  # Invert gripper
        ]
    )
    return bridge_act


class WidowXGym(gym.Env):
    """
    A Gym environment for the WidowX controller provided by:
    https://github.com/rail-berkeley/bridge_data_robot
    Needed to use Gym wrappers.
    """

    def __init__(
        self,
        widowx_client: WidowXClient,
        blocking: bool = True,
        sticky_gripper_num_steps: int = 1,
        resize_shape: Optional[Tuple[int, int]] = None,
    ):
        self.widowx_client = widowx_client
        self.blocking = blocking
        self.resize_shape = resize_shape
        image_shape = (*resize_shape, 3) if resize_shape is not None else (480, 640, 3)

        self.observation_space = gym.spaces.Dict(
            dict(
                image=gym.spaces.Dict(
                    dict(
                        agent=gym.spaces.Box(shape=image_shape, dtype=np.uint8, low=0, high=255),
                    )
                ),
                state=gym.spaces.Dict(
                    {
                        StateEncoding.EE_POS: gym.spaces.Box(shape=(3,), low=-np.inf, high=np.inf, dtype=np.float32),
                        StateEncoding.EE_EULER: gym.spaces.Box(shape=(3,), low=-np.inf, high=np.inf, dtype=np.float32),
                        StateEncoding.GRIPPER: gym.spaces.Box(shape=(1,), low=-np.inf, high=np.inf, dtype=np.float32),
                    }
                ),
            )
        )

        self.action_space = gym.spaces.Dict(
            dict(
                achieved_delta=gym.spaces.Dict(
                    {
                        StateEncoding.EE_POS: gym.spaces.Box(shape=(3,), low=-np.inf, high=np.inf, dtype=np.float32),
                        StateEncoding.EE_EULER: gym.spaces.Box(shape=(3,), low=-np.inf, high=np.inf, dtype=np.float32),
                    }
                ),
                desired_delta=gym.spaces.Dict(
                    {
                        StateEncoding.EE_POS: gym.spaces.Box(shape=(3,), low=-np.inf, high=np.inf, dtype=np.float32),
                        StateEncoding.EE_EULER: gym.spaces.Box(shape=(3,), low=-np.inf, high=np.inf, dtype=np.float32),
                    }
                ),
                desired_absolute=gym.spaces.Dict(
                    {StateEncoding.GRIPPER: gym.spaces.Box(shape=(1,), low=-np.inf, high=np.inf, dtype=np.float32)}
                ),
            )
        )

        self.sticky_gripper_num_steps = sticky_gripper_num_steps
        self.is_gripper_closed = False
        self.num_consecutive_gripper_change_actions = 0

    def step(self, action):
        # sticky gripper logic
        trans_action = convert_act(action)  # transform action back into act format used in bridge env
        print(trans_action)
        if (trans_action[-1] < 0.5) != self.is_gripper_closed:
            self.num_consecutive_gripper_change_actions += 1
        else:
            self.num_consecutive_gripper_change_actions = 0

        if self.num_consecutive_gripper_change_actions >= self.sticky_gripper_num_steps:
            self.is_gripper_closed = not self.is_gripper_closed
            self.num_consecutive_gripper_change_actions = 0
        trans_action[-1] = 0.0 if self.is_gripper_closed else 1.0
        self.widowx_client.step_action(trans_action, blocking=self.blocking)

        raw_obs = self.widowx_client.get_observation()

        truncated = False
        if raw_obs is None:
            # this indicates a loss of connection with the server
            # due to an exception in the last step so end the trajectory
            truncated = True
            obs = null_obs()  # obs with all zeros
        else:
            obs = convert_obs(raw_obs, resize_shape=self.resize_shape)

        # Write the goal in
        return obs, 0, False, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.widowx_client.reset()

        self.is_gripper_closed = False
        self.num_consecutive_gripper_change_actions = 0

        raw_obs = wait_for_obs(self.widowx_client)
        obs = convert_obs(raw_obs, resize_shape=self.resize_shape)

        return obs, {}
