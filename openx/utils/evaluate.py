from typing import Callable, Dict

import gymnasium as gym
import jax
import numpy as np


def eval_policy(
    env: gym.Env,
    predict: Callable,
    rng: jax.random.PRNGKey,
    num_ep: int = 10,
) -> Dict:
    if not isinstance(env, gym.vector.VectorEnv):
        env = gym.vector.SyncVectorEnv(lambda: env)
    num_envs = env.num_envs

    rewards, lengths, successes = [], [], []
    ep_length = np.zeros((num_envs,), dtype=np.int32)
    ep_reward = np.zeros((num_envs,), dtype=np.float32)
    ep_success = np.zeros((num_envs,), dtype=np.bool_)
    obs, info = env.reset()
    steps = 0

    while len(rewards) < num_ep:
        steps += 1
        rng = jax.random.fold_in(rng, steps)
        action = predict(obs, rng=rng)
        action = np.asarray(action)  # Must convert away from jax tensor.
        obs, reward, done, trunc, info = env.step(action)
        ep_reward += reward
        ep_length += 1
        if "success" in info:
            ep_success = np.logical_or(ep_success, info["success"])

        # Determine if we are done.
        for i in range(num_envs):
            if done[i] or trunc[i]:
                rewards.append(ep_reward[i])
                lengths.append(ep_length[i])
                # Need to manually check for success
                successes.append(ep_success[i] or info["final_info"][i]["success"])
                ep_reward[i] = 0.0
                ep_length[i] = 0
                ep_success[i] = False

    eval_metrics = dict(reward=np.mean(rewards), success=np.mean(successes), length=np.mean(lengths))
    print(eval_metrics)
    return eval_metrics
