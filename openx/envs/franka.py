import gymnasium as gym
import numpy as np
import robots

from openx.data.utils import StateEncoding


def convert_obs(obs):
    obs = dict(
        image=dict(
            agent=obs["agent_image"],
            wrist=obs["wrist_image"],
        ),
        # For consistency resize using dlimp
        # new_obs["image_wrist"] = dl.transforms.resize_image(obs["wrist_image"], size=(128, 128)).numpy()
        state={
            StateEncoding.EE_POS: obs["state"]["ee_pos"],
            StateEncoding.EE_QUAT: obs["state"]["ee_quat"],
            StateEncoding.GRIPPER: obs["state"]["gripper_pos"],
        },
    )
    return obs


def convert_act(act):
    """
    convert the action dict used in openX back to action used in original bridge data robot env
    return: act, shape(7,)
    """
    ee_key = next(k for k in ("achieved_delta", "desired_delta") if k in act)
    # TODO: print out gripper actions?
    bridge_act = np.concatenate(
        [
            act[ee_key][StateEncoding.EE_POS],
            act[ee_key][StateEncoding.EE_EULER],
            np.clip(
                2 * act["desired_absolute"][StateEncoding.GRIPPER] - 1, a_min=-1, a_max=1
            ),  # Rescale gripper back to [-1, 1]
        ]
    )
    return bridge_act


class FrankaEnv(gym.Env):
    def __init__(self, **robot_config):
        super().__init__()
        self.env = robots.RobotEnv(**robot_config)
        image_shape = self.env.observation_space["agent_image"].shape

        self.observation_space = gym.spaces.Dict(
            dict(
                image=gym.spaces.Dict(
                    dict(
                        agent=gym.spaces.Box(shape=image_shape, dtype=np.uint8, low=0, high=255),
                        wrist=gym.spaces.Box(shape=image_shape, dtype=np.uint8, low=0, high=255),
                    )
                ),
                state=gym.spaces.Dict(
                    {
                        StateEncoding.EE_POS: gym.spaces.Box(shape=(3,), low=-np.inf, high=np.inf, dtype=np.float32),
                        StateEncoding.EE_QUAT: gym.spaces.Box(shape=(4,), low=-np.inf, high=np.inf, dtype=np.float32),
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

    def step(self, action):
        # We need to reverse the gripper action
        action = convert_act(action)
        obs, reward, done, truncated, info = self.env.step(action)

        return convert_obs(obs), reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        return convert_obs(obs), info
