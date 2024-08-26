import json
import os
from functools import partial

import flax
import gymnasium as gym
import jax
import numpy as np
import tensorflow as tf
from absl import app, flags
from jax.experimental import compilation_cache
from ml_collections import ConfigDict
from orbax import checkpoint

from openx.envs.wrappers import wrap_env
from openx.utils.evaluate import eval_policy
from openx.utils.spec import ModuleSpec, add_kwarg, recursively_instantiate

FLAGS = flags.FLAGS
flags.DEFINE_string("path", None, "Path to save logs and checkpoints.")
flags.DEFINE_string("checkpoint_step", None, "Checkpoint step to load.")
flags.DEFINE_integer("num_ep", None, "Number of eval ep.")
flags.DEFINE_integer("num_workers", None, "Number of eval workers.")

def main(_):
    # Initialize experimental jax compilation cache
    compilation_cache.compilation_cache.set_cache_dir(os.path.expanduser("~/.jax_compilation_cache"))

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # Load the example batch
    with tf.io.gfile.GFile(tf.io.gfile.join(FLAGS.path, "example_batch.msgpack"), "rb") as f:
        example_batch = flax.serialization.msgpack_restore(f.read())

    # Load the dataset statistics
    with tf.io.gfile.GFile(tf.io.gfile.join(FLAGS.path, "dataset_statistics.json"), "r") as f:
        dataset_statistics = json.load(f)

        def _convert_to_numpy(x):
            return {
                k: _convert_to_numpy(v) if isinstance(v, dict) else np.array(v, dtype=np.float32) for k, v in x.items()
            }

        dataset_statistics = _convert_to_numpy(dataset_statistics)

    # Load the config
    with tf.io.gfile.GFile(tf.io.gfile.join(FLAGS.path, "config.json"), "r") as f:
        config = json.load(f)
        config = ConfigDict(config)

    action_horizon, action_dim = example_batch["action"].shape[-2:]

    # Instantiate the model
    model_config = config.model.to_dict()
    # A bit of a hack for now to deliver the action_horizon and action_dim to the action_head
    add_kwarg(model_config, "action_head.action_horizon", action_horizon)
    add_kwarg(model_config, "action_head.action_dim", action_dim)
    model = recursively_instantiate(model_config)

    rng = jax.random.PRNGKey(config.seed)

    shapes = jax.eval_shape(partial(model.init, train=False), rng, example_batch)
    checkpointer = checkpoint.CheckpointManager(FLAGS.path, checkpoint.PyTreeCheckpointer())
    step = FLAGS.checkpoint_step if FLAGS.checkpoint_step is not None else checkpointer.latest_step()
    params = checkpointer.restore(step, shapes)

    ### Define the Predict Function ###
    @jax.jit
    def predict(params, obs, rng):
        batch = dict(observation=obs)
        action = model.apply(params, batch, rngs=dict(dropout=rng), train=False, method=model.predict)
        return action

    ### Setup Eval Envs ###
    envs = dict()
    if config.get("envs", None) is not None and len(config.envs) > 0:
        structure = config.structure.to_dict()
        n_obs, n_action = config.dataloader.n_obs, config.dataloader.n_action
        scale_range = config.dataloader.augment_kwargs.get("scale_range", None)

        def _make_env(fn, stats):
            env = fn()
            env = wrap_env(
                env,
                structure=structure,
                dataset_statistics=stats,
                n_obs=n_obs,
                n_action=n_action,
                exec_horizon=max(1, n_action // 2),
                scale_range=scale_range,
            )
            return env

        n_eval_proc = FLAGS.num_workers if FLAGS.num_workers else 1
        for env_name, env_spec in config.envs.to_dict().items():
            env_fn = partial(_make_env, fn=ModuleSpec.instantiate(env_spec), stats=dataset_statistics[env_name])
            envs[env_name] = gym.vector.AsyncVectorEnv(
                [env_fn for _ in range(n_eval_proc)], context="spawn", shared_memory=True
            )

        # Evaluate the model
        num_ep = FLAGS.num_ep if FLAGS.num_ep else config.eval_ep
        for env_name, env in envs.items():
            eval_metrics = eval_policy(env, partial(predict, params), rng, num_ep=num_ep)
            print(env_name, eval_metrics)


if __name__ == "__main__":
    app.run(main)
