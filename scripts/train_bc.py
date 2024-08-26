import datetime
import json
import os
from collections import defaultdict
from functools import partial

import flax
import gymnasium as gym
import jax
import numpy as np
import optax
import orbax
import tensorflow as tf
import tqdm
from absl import app, flags
from flax.training import orbax_utils, train_state
from jax.experimental import compilation_cache, multihost_utils
from ml_collections import config_flags

import wandb
from openx.data.dataloader import make_dataloader
from openx.envs.wrappers import wrap_env
from openx.utils.evaluate import eval_policy
from openx.utils.logger import DummyLogger, Logger, Timer
from openx.utils.spec import ModuleSpec, add_kwarg, recursively_instantiate

FLAGS = flags.FLAGS
flags.DEFINE_string("path", "/tmp/test/", "Path to save logs and checkpoints.")
flags.DEFINE_string("name", "train_bc", "Name of the experiment")
flags.DEFINE_string("project", "openx", "WandB project to save logs to.")
flags.DEFINE_bool("debug", False, "Whether or not to enable debug mode.")
# Always lock the config to avoid subtle bugs
config_flags.DEFINE_config_file(
    "config", None, "File path to the training hyperparameter configuration.", lock_config=True
)


class TrainState(train_state.TrainState):
    rng: jax.random.PRNGKey


def main(_):
    # Initialize experimental jax compilation cache
    compilation_cache.compilation_cache.set_cache_dir(os.path.expanduser("~/.jax_compilation_cache"))
    assert FLAGS.config.dataloader.batch_size % jax.device_count() == 0

    # Define Shardings
    mesh = jax.sharding.Mesh(jax.devices(), axis_names="batch")
    dp_spec = jax.sharding.PartitionSpec("batch")
    dp_sharding = jax.sharding.NamedSharding(mesh, dp_spec)
    rep_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    def shard(batch):
        return multihost_utils.host_local_array_to_global_array(batch, mesh, dp_spec)

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # make sure each process loads different data
    tf.random.set_seed(FLAGS.config.seed + jax.process_index())

    # Create the dataloader
    train_dataset, val_datasets, dataset_statistics, _ = make_dataloader(
        **FLAGS.config.dataloader.to_dict(), structure=FLAGS.config.structure.to_dict(), split_for_jax=True
    )

    # Create the data iterators
    # Note that we directly get the numpy representation from tensorflow to avoid a copy.
    train_iterator = map(
        shard,
        map(
            lambda b: jax.tree_util.tree_map(lambda x: x._numpy(), b),
            train_dataset.prefetch(0),  # Set to Zero
        ),
    )
    val_iterators = {
        p: map(shard, map(lambda b: jax.tree_util.tree_map(lambda x: x._numpy(), b), ds))
        for p, ds in val_datasets.items()
    }

    # Deque the first batch to use as an example for instantiating the model
    example_batch = jax.tree_map(lambda x: x[:1], multihost_utils.process_allgather(next(train_iterator)))
    action_horizon, action_dim = example_batch["action"].shape[-2:]

    # Instantiate the model
    model_config = FLAGS.config.model.to_dict()
    # A bit of a hack for now to deliver the action_horizon and action_dim to the action_head
    add_kwarg(model_config, "action_head.action_horizon", action_horizon)
    add_kwarg(model_config, "action_head.action_dim", action_dim)
    model = recursively_instantiate(model_config)

    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, init_rng = jax.random.split(rng)

    params = jax.jit(partial(model.init, train=False))(init_rng, example_batch)

    lr_schedule = ModuleSpec.instantiate(FLAGS.config.lr_schedule)()
    tx = ModuleSpec.instantiate(FLAGS.config.optimizer)
    if tx.func is optax.adamw:  # A bit of a hack for now to properly decay params
        decay_mask = jax.tree_util.tree_map_with_path(lambda path, _: "kernel" in jax.tree_util.keystr(path), params)
        tx = partial(tx, mask=decay_mask)
    tx = tx(learning_rate=lr_schedule)  # Finally create the optimizer
    if "clip_gradient" in FLAGS.config and FLAGS.config.clip_gradient is not None:
        tx = optax.chain(optax.clip_by_global_norm(FLAGS.config.clip_gradient), tx)
    state = TrainState.create(apply_fn=partial(model.apply, method=model.loss), params=params, tx=tx, rng=rng)

    ### Define the Train Step ###
    @partial(
        jax.jit,
        in_shardings=(rep_sharding, dp_sharding),
        out_shardings=(rep_sharding, rep_sharding),
        donate_argnums=0,
    )
    def train_step(state, batch):
        rng, dropout_key = jax.random.split(state.rng)
        loss, grads = jax.value_and_grad(state.apply_fn)(
            state.params, batch, train=True, rngs=dict(dropout=dropout_key)
        )
        info = dict(loss=loss, lr=lr_schedule(state.step))
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    ### Define the Val Step ###
    @partial(
        jax.jit,
        in_shardings=(rep_sharding, dp_sharding),
        out_shardings=(rep_sharding, rep_sharding),
        donate_argnums=0,
    )
    def val_step(state, batch):
        _, dropout_key = jax.random.split(state.rng)
        return model.apply(
            state.params, batch, rngs=dict(dropout=dropout_key), train=False, method=model.loss_and_prediction_mse
        )

    ### Define the Predict Function ###
    @jax.jit
    def predict(state, obs, rng):
        batch = dict(observation=obs)
        action = model.apply(state.params, batch, rngs=dict(dropout=rng), train=False, method=model.predict)
        return action

    ### Setup Eval Envs ###
    envs = dict()
    if FLAGS.config.get("envs", None) is not None and len(FLAGS.config.envs) > 0:
        structure = FLAGS.config.structure.to_dict()
        n_obs, n_action = FLAGS.config.dataloader.n_obs, FLAGS.config.dataloader.n_action
        scale_range = FLAGS.config.dataloader.augment_kwargs.get("scale_range", None)

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

        for env_name, env_spec in FLAGS.config.envs.to_dict().items():
            env_fn = partial(_make_env, fn=ModuleSpec.instantiate(env_spec), stats=dataset_statistics[env_name])
            envs[env_name] = gym.vector.AsyncVectorEnv(
                [env_fn for _ in range(FLAGS.config.n_eval_proc)], context="spawn", shared_memory=True
            )

    ### Broadcast name across all hosts ###
    name = "{name}_{time}".format(name=FLAGS.name, time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    name = multihost_utils.broadcast_one_to_all(np.array([ord(c) for c in name], dtype=np.uint8))
    name = "".join([chr(c) for c in name])

    ### Init Checkpointing ###
    save_path = tf.io.gfile.join(FLAGS.path, name)
    if not FLAGS.debug:
        state_checkpointer = orbax.checkpoint.CheckpointManager(
            tf.io.gfile.join(save_path, "state"),
            orbax.checkpoint.PyTreeCheckpointer(),
            options=orbax.checkpoint.CheckpointManagerOptions(max_to_keep=1, create=True),
        )
        weights_checkpointer = orbax.checkpoint.CheckpointManager(save_path, orbax.checkpoint.PyTreeCheckpointer())

    ### Worker Saves Statistics, Configs, ExBatch ###
    if jax.process_index() == 0 and not FLAGS.debug:
        # Save the example batch
        example_batch_path = tf.io.gfile.join(save_path, "example_batch.msgpack")
        with tf.io.gfile.GFile(example_batch_path, "wb") as f:
            f.write(flax.serialization.msgpack_serialize(example_batch))

        # Save the dataset statistics
        dataset_statistics_path = tf.io.gfile.join(save_path, "dataset_statistics.json")
        with tf.io.gfile.GFile(dataset_statistics_path, "w") as f:
            json.dump(
                jax.tree_map(lambda x: x.tolist() if isinstance(x, np.ndarray) else x, dataset_statistics), f, indent=4
            )

        # Save the config
        config_path = tf.io.gfile.join(save_path, "config.json")
        with tf.io.gfile.GFile(config_path, "w") as f:
            json.dump(FLAGS.config.to_dict(), f, indent=4)

        # Init wandb logging
        wandb.init(
            config=FLAGS.config.to_dict(),
            project=FLAGS.project,
            name=name,
            mode="offline" if FLAGS.debug else "online",
        )

    if jax.process_index() == 0:
        # Init Logging
        logger = Logger(save_path, writers=() if FLAGS.debug else ("csv",))
    else:
        logger = DummyLogger()
    timer = Timer()

    # Training constants
    train_metrics = defaultdict(list)
    for i in tqdm.tqdm(range(FLAGS.config.steps), total=FLAGS.config.steps, dynamic_ncols=True):
        with timer("dataset"):
            batch = next(train_iterator)

        with timer("train"):
            state, info = train_step(state, batch)
            for k, v in info.items():
                train_metrics[k].append(v)

        step = i + 1
        if step % FLAGS.config.log_freq == 0:
            # Log training loss and timing
            logger.update(train_metrics, prefix="train")
            logger.update(timer.times, prefix="time")

            logger.dump(step=step, eval=False)
            train_metrics = defaultdict(list)
            timer.reset()

        if step % FLAGS.config.val_freq == 0:
            # Run evaluation
            val_metrics = defaultdict(list)
            with timer("val"):
                for p, val_iterator in val_iterators.items():
                    p = p.replace("/", "-")  # Remove the '/' for logger
                    for _ in tqdm.tqdm(range(FLAGS.config.val_steps), total=FLAGS.config.val_steps):
                        batch = next(val_iterator)
                        val_loss, val_mse = val_step(state, batch)
                        val_metrics[p + "/loss"].append(val_loss)
                        val_metrics[p + "/mse"].append(val_mse)

            logger.update(val_metrics, prefix="val")
            logger.dump(step=step, eval=True)

        if step % FLAGS.config.eval_freq == 0:
            for env_name, env in envs.items():
                with timer("eval/" + env_name):
                    eval_metrics = eval_policy(env, partial(predict, state), state.rng, num_ep=FLAGS.config.eval_ep)
                    logger.update(eval_metrics, prefix="eval/" + env_name)
            # Dump the logger with eval metrics
            logger.dump(step=step, eval=True)

        if step % FLAGS.config.save_freq == 0 and not FLAGS.debug:
            # save the train state.
            with timer("save"):
                state_checkpointer.save(
                    step, state, save_kwargs=dict(save_args=orbax_utils.save_args_from_target(state))
                )
                weights_checkpointer.save(
                    step, state.params, save_kwargs=dict(save_args=orbax_utils.save_args_from_target(state.params))
                )


if __name__ == "__main__":
    app.run(main)
