from collections import deque
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import tensorflow as tf

from openx.data.core import filter_by_structure, filter_dataset_statistics_by_structure
from openx.data.transforms import _center_bbox, _normalize, _unnormalize


def space_stack(space: gym.Space, repeat: int):
    """
    Creates new Gym space that represents the original observation/action space
    repeated `repeat` times.
    """

    if isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(
            low=np.repeat(space.low[None], repeat, axis=0),
            high=np.repeat(space.high[None], repeat, axis=0),
            dtype=space.dtype,
        )
    elif isinstance(space, gym.spaces.Discrete):
        return gym.spaces.MultiDiscrete([space.n] * repeat)
    elif isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict({k: space_stack(v, repeat) for k, v in space.spaces.items()})
    else:
        raise ValueError(f"Space {space} is not supported by Octo Gym wrappers.")


def convert_to_space(space):
    if isinstance(space, gym.Space):
        return space
    elif isinstance(space, dict):
        return gym.spaces.Dict({k: convert_to_space(v) for k, v in space.items()})
    elif isinstance(space, np.ndarray):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=space.shape, dtype=space.dtype)
    # For now, don't handle discrete spaces.
    else:
        raise ValueError("Invalid input passed to `convert_to_space`: " + str(type(space)))


class HistoryWrapper(gym.Wrapper):
    """
    Accumulates the observation history into `horizon` size chunks. If the length of the history
    is less than the length of the horizon, we pad the history to the full horizon length.
    A `timestep_pad_mask` key is added to the final observation dictionary that denotes which timesteps
    are padding.
    """

    def __init__(self, env: gym.Env, horizon: int, mask_keys: Tuple = ("state",)):
        super().__init__(env)
        self.horizon = horizon
        self.history = deque(maxlen=self.horizon)
        self.observation_space = space_stack(self.env.observation_space, self.horizon)
        self.mask_keys = mask_keys

    def _get_obs(self):
        return tf.nest.map_structure(lambda *args: np.stack(args, dtype=args[0].dtype), *self.history)

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        for mask_key in self.mask_keys:
            if mask_key in self.history[-1]:
                self.history[-1][mask_key][:] = 0
        self.history.append(obs)
        assert len(self.history) == self.horizon
        return self._get_obs(), reward, done, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.history.extend([obs] * self.horizon)
        return self._get_obs(), info


class RHCWrapper(gym.Wrapper):
    """
    Performs receding horizon control. The policy returns `action_horizon` actions and
    we execute `exec_horizon` of them.
    """

    def __init__(self, env: gym.Env, policy_horizon: int, exec_horizon: int):
        super().__init__(env)
        assert policy_horizon >= exec_horizon
        self.exec_horizon = exec_horizon
        self.action_space = space_stack(self.env.action_space, policy_horizon)

    def step(self, actions):
        if self.exec_horizon == 1 and len(actions.shape) == 1:
            actions = actions[None]
        assert len(actions) >= self.exec_horizon
        total_reward = 0
        infos = []

        for i in range(self.exec_horizon):
            obs, reward, done, trunc, info = self.env.step(actions[i])
            total_reward += reward
            infos.append(info)

            if done or trunc:
                break

        infos = {k: [dic[k] for dic in infos] for k in infos[0]}
        # Explicitly reduce success so we log it.
        if "success" in infos:
            infos["success"] = any(infos["success"])
        return obs, total_reward, done, trunc, infos


class StructureWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        structure: Dict,
    ):
        super().__init__(env)
        self.structure = structure
        self.observation_space = convert_to_space(
            filter_by_structure(self.env.observation_space, structure["observation"])
        )
        self.action_space = convert_to_space(filter_by_structure(self.env.action_space, structure["action"]))

    def _standardize_structure(self, obs):
        # Select only the relevant keys
        return filter_by_structure(obs, self.structure["observation"])

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        return self._standardize_structure(obs), reward, done, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._standardize_structure(obs), info


def _resize_images(obs: Dict, structure: Dict, scale_range: Optional[List] = None):
    # Ok yes this function does modify in place, but **shrug** for now I'm tired.
    if "image" in structure:
        output_image_obs = dict()
        for k, shape in structure["image"].items():
            imgs = obs["image"][k]
            center_bbox = _center_bbox(imgs.shape, shape, scale_range=scale_range)
            imgs = tf.image.crop_to_bounding_box(imgs, *center_bbox)
            # TODO: test adjusting the image jpeg qualtiy to account for dataset compression.
            # imgs = tf.stack([tf.image.adjust_jpeg_quality(img, 95) for img in tf.unstack(imgs, axis=0)], axis=0)
            imgs = tf.image.convert_image_dtype(imgs, dtype=tf.float32)
            imgs = tf.image.resize(imgs, shape)  # Resize using bilinear as done in the random crop op!
            imgs = imgs._numpy()  # Convert back to numpy
            output_image_obs[k] = imgs
        obs["image"] = output_image_obs
    return obs


class ResizeImageWrapper(gym.Wrapper):
    def __init__(self, env, structure: Dict, scale_range: Optional[Tuple[float, float]] = None):
        super().__init__(env)
        self.structure = structure
        self.scale_range = scale_range
        assert isinstance(self.env.observation_space, gym.spaces.Dict)
        spaces = {k: v for k, v in self.env.observation_space.spaces.items() if k != "image"}
        if "image" in self.env.observation_space.spaces:
            image_spaces = {
                k: gym.spaces.Box(
                    shape=(*self.structure["observation"]["image"][k], 3), low=0, high=1, dtype=np.float32
                )
                for k in self.env.observation_space["image"].spaces.keys()
            }
            spaces["image"] = gym.spaces.Dict(image_spaces)
        self.observation_space = gym.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        obs = _resize_images(obs, self.structure["observation"], scale_range=self.scale_range)
        return obs, reward, done, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = _resize_images(obs, self.structure["observation"], scale_range=self.scale_range)
        return obs, info


class NormalizationWrapper(gym.Wrapper):
    def __init__(self, env, structure, dataset_statistics):
        super().__init__(env)
        self.structure = structure
        self.dataset_statistics = filter_dataset_statistics_by_structure(dataset_statistics, structure)

    def _normalize_obs(self, obs):
        if "state" in obs:
            tf.nest.assert_same_structure(self.dataset_statistics["mean"]["state"], obs["state"])
            obs["state"] = tf.nest.map_structure(
                _normalize,
                obs["state"],
                self.structure["observation"]["state"],
                self.dataset_statistics["mean"]["state"],
                self.dataset_statistics["std"]["state"],
                self.dataset_statistics["min"]["state"],
                self.dataset_statistics["max"]["state"],
            )
        return obs

    def step(self, action):
        # Unnormalize the actions
        tf.nest.assert_same_structure(self.dataset_statistics["mean"]["action"], action)
        action = tf.nest.map_structure(
            _unnormalize,
            action,
            self.structure["action"],
            self.dataset_statistics["mean"]["action"],
            self.dataset_statistics["std"]["action"],
            self.dataset_statistics["min"]["action"],
            self.dataset_statistics["max"]["action"],
        )
        obs, reward, done, trunc, info = self.env.step(action)
        return self._normalize_obs(obs), reward, done, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._normalize_obs(obs), info


class ConcatenationWrapper(gym.Wrapper):
    """
    Concatenates the states and actions
    """

    def __init__(self, env, structure):
        super().__init__(env)
        # Edit the observation space
        spaces = {k: v for k, v in self.env.observation_space.spaces.items() if k != "state"}
        if "state" in self.env.observation_space.spaces:
            state_spaces = tf.nest.flatten(self.env.observation_space["state"].spaces)
            state_space = gym.spaces.Box(
                low=np.concatenate([space.low for space in state_spaces]),
                high=np.concatenate([space.high for space in state_spaces]),
                dtype=np.float32,
            )
            spaces["state"] = state_space
        self.observation_space = gym.spaces.Dict(spaces)

        # Keep only the action types in the structure.
        sample = self.env.action_space.sample()
        sample = filter_by_structure(sample, structure["action"])
        action_lengths = list(tf.nest.flatten(tf.nest.map_structure(lambda x: x.shape[0], sample)))
        self.action_space = gym.spaces.Box(shape=(sum(action_lengths),), low=-np.inf, high=np.inf, dtype=np.float32)
        self.action_split_points = np.cumsum(action_lengths)[:-1]  # remove the last value.
        self.sample = sample

    def _concatenate_obs(self, obs):
        if "state" in obs:
            obs["state"] = np.concatenate(tf.nest.flatten(obs["state"]), axis=-1)
        return obs

    def step(self, action):
        action = np.split(action, self.action_split_points)
        action = tf.nest.pack_sequence_as(self.sample, action)
        obs, reward, done, trunc, info = self.env.step(action)
        return self._concatenate_obs(obs), reward, done, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._concatenate_obs(obs), info


class NumpyWrapper(gym.Wrapper):
    """
    Converts all remaining tf arrays to numpy
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def numpy_obs(self, obs):
        return tf.nest.map_structure(lambda x: x._numpy() if isinstance(x, tf.Tensor) else x, obs)

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        return self.numpy_obs(obs), reward, done, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.numpy_obs(obs), info


def wrap_env(
    env,
    structure: Dict,
    dataset_statistics: Optional[Dict] = None,
    n_obs: int = 1,
    n_action: int = 1,
    exec_horizon: int = 1,
    scale_range: Optional[Tuple[float, float]] = None,
):
    env = StructureWrapper(env, structure)
    if dataset_statistics is not None:
        env = NormalizationWrapper(env, structure, dataset_statistics)
    env = ConcatenationWrapper(env, structure)
    env = ResizeImageWrapper(env, structure, scale_range=scale_range)
    if n_obs is not None:
        env = HistoryWrapper(env, horizon=n_obs)
    if n_action is not None:
        env = RHCWrapper(env, n_action, exec_horizon)
    return env


def preprocess_goal(
    goal,
    structure: Dict,
    dataset_statistics: Optional[Dict] = None,
    scale_range: Optional[Tuple[float, float]] = None,
):
    # Processes a goal dictionary to be passed into a model
    goal = filter_by_structure(goal, structure["observation"])
    # Normalize and then concatenate state
    if "state" in goal:
        assert dataset_statistics is not None
        dataset_statistics = filter_dataset_statistics_by_structure(dataset_statistics, structure)
        goal["state"] = tf.nest.map_structure(
            _normalize,
            goal["state"],
            structure["observation"]["state"],
            dataset_statistics["mean"]["state"],
            dataset_statistics["std"]["state"],
            dataset_statistics["min"]["state"],
            dataset_statistics["max"]["state"],
        )
        goal["state"] = np.concatenate(tf.nest.flatten(goal["state"]), axis=-1)
    # Resize images
    goal = _resize_images(goal, structure["observation"], scale_range=scale_range)
    # Add the temporal dimension
    goal = tf.nest.map_structure(lambda x: x[None], goal)
    return goal
