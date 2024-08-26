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
from jax import numpy as jnp
from jax.experimental import compilation_cache, multihost_utils
from ml_collections import ConfigDict, config_flags
from orbax import checkpoint

import wandb
from openx.data.dataloader import make_dataloader
from openx.envs.wrappers import wrap_env
from openx.utils.logger import DummyLogger, Logger, Timer
from openx.utils.spec import ModuleSpec, add_kwarg, recursively_instantiate

FLAGS = flags.FLAGS
flags.DEFINE_string("path", "/tmp/test", "Path to save logs and checkpoints.")
flags.DEFINE_string("checkpoint_path", None, "Path to the reference model checkpoint")
flags.DEFINE_string("checkpoint_step", None, "Checkpoint step or none")
flags.DEFINE_string("name", "doremi", "Name of the experiment")
flags.DEFINE_string("project", "openx", "WandB project to save logs to.")
flags.DEFINE_bool("debug", False, "Whether or not to enable debug mode.")
# Always lock the config to avoid subtle bugs
config_flags.DEFINE_config_file(
    "config", None, "File path to the training hyperparameter configuration.", lock_config=True
)


class DoremiTrainState(train_state.TrainState):
    rng: jax.random.PRNGKey
    alpha: jax.Array
    average_alpha: jax.Array


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
    # First fetch the dataset statistics from the reference model
    with tf.io.gfile.GFile(tf.io.gfile.join(FLAGS.checkpoint_path, "dataset_statistics.json"), "r") as f:
        dataset_statistics = json.load(f)

        def _convert_to_numpy(x):
            return {
                k: _convert_to_numpy(v) if isinstance(v, dict) else np.array(v, dtype=np.float32) for k, v in x.items()
            }

        dataset_statistics = _convert_to_numpy(dataset_statistics)
    # Then construct the dataloader
    train_dataset, _, dataset_statistics, dataset_ids = make_dataloader(
        **FLAGS.config.dataloader.to_dict(),
        structure=FLAGS.config.structure.to_dict(),
        dataset_statistics=dataset_statistics,
        split_for_jax=True,
    )

    # Create the data iterators
    # Note that we directly get the numpy representation from tensorflow to avoid a copy.
    train_iterator = map(
        shard,
        map(
            lambda b: jax.tree_util.tree_map(lambda x: x.numpy(), b),
            train_dataset.prefetch(0),  # Set to Zero
        ),
    )

    # Deque the first batch to use as an example for instantiating the model
    example_batch = jax.tree_map(lambda x: x[:1], multihost_utils.process_allgather(next(train_iterator)))
    action_horizon, action_dim = example_batch["action"].shape[-2:]

    # Load the reference model
    with tf.io.gfile.GFile(tf.io.gfile.join(FLAGS.checkpoint_path, "config.json"), "r") as f:
        ref_config = json.load(f)
        ref_config = ConfigDict(ref_config)

    # TODO: Figure out if we can do checks to dataset. These currently don't work becuase of enums.
    # assert ref_config.structure.observation.to_dict() == FLAGS.config.structure.observation.to_dict()
    # assert ref_config.structure.action.to_dict() == FLAGS.config.structure.action.to_dict()

    ref_model_config = ref_config.model.to_dict()
    add_kwarg(ref_model_config, "action_head.action_horizon", action_horizon)
    add_kwarg(ref_model_config, "action_head.action_dim", action_dim)
    ref_model = recursively_instantiate(ref_model_config)

    rng = jax.random.PRNGKey(FLAGS.config.seed)

    shapes = jax.eval_shape(partial(ref_model.init, train=False), rng, example_batch)
    ref_checkpointer = checkpoint.CheckpointManager(FLAGS.checkpoint_path, checkpoint.PyTreeCheckpointer())
    step = FLAGS.checkpoint_step if FLAGS.checkpoint_step is not None else ref_checkpointer.latest_step()
    ref_params = ref_checkpointer.restore(step, shapes)

    # Now create the Proxy model
    proxy_model_config = FLAGS.config.model.to_dict()
    # A bit of a hack for now to deliver the action_horizon and action_dim to the action_head
    add_kwarg(proxy_model_config, "action_head.action_horizon", action_horizon)
    add_kwarg(proxy_model_config, "action_head.action_dim", action_dim)
    proxy_model = recursively_instantiate(proxy_model_config)

    rng, init_rng = jax.random.split(rng)

    proxy_params = jax.jit(partial(proxy_model.init, train=False))(init_rng, example_batch)
    lr_schedule = ModuleSpec.instantiate(FLAGS.config.lr_schedule)()
    tx = ModuleSpec.instantiate(FLAGS.config.optimizer)
    if tx.func is optax.adamw:  # A bit of a hack for now to properly decay params
        decay_mask = jax.tree_util.tree_map_with_path(
            lambda path, _: "kernel" in jax.tree_util.keystr(path), proxy_params
        )
        tx = partial(tx, mask=decay_mask)
    tx = tx(learning_rate=lr_schedule)  # Finally create the optimizer
    if "clip_gradient" in FLAGS.config and FLAGS.config.clip_gradient is not None:
        tx = optax.chain(optax.clip_by_global_norm(FLAGS.config.clip_gradient), tx)

    # Initialize the DoReMi weights to be the size of each dataset.
    if FLAGS.config.domain_key == "dataset_id":
        id_to_dataset = {v: k for k, v in dataset_ids.items()}
        # TODO: update this to be the weights created by the dataloader!
        initial_alpha = jnp.array(
            [dataset_statistics[id_to_dataset[i]]["num_steps"] for i in range(len(dataset_ids))], dtype=jnp.float32
        )
    elif FLAGS.config.initial_alpha is not None:
        initial_alpha = jnp.array(FLAGS.config.initial_alpha).astype(jnp.float32)
    else:
        initial_alpha = jnp.ones(FLAGS.config.num_domains, dtype=jnp.float32)
    initial_alpha = initial_alpha / jnp.sum(initial_alpha)  # Make sure it sums to 1.
    state = DoremiTrainState.create(
        apply_fn=partial(proxy_model.apply, method=proxy_model.loss, reduce=False),
        params=proxy_params,
        tx=tx,
        rng=rng,
        alpha=initial_alpha,
        average_alpha=initial_alpha,
    )

    ### Define the Train Step ###

    # List out the parameters
    smoothing = FLAGS.config.smoothing
    domain_key = FLAGS.config.domain_key
    num_domains = FLAGS.config.get("num_domains", None) if domain_key != "dataset_id" else len(dataset_ids)
    domain_weight_step_size = FLAGS.config.domain_weight_step_size

    def _compute_per_domain_losses(losses, domains):
        one_hot_domains = jax.nn.one_hot(domains, num_domains, axis=0)  # (D, B)
        per_domain_losses = jnp.dot(one_hot_domains, losses)  # (D, B) dot (B,) -> D
        # count the number of losses for each domain
        norm = jnp.dot(one_hot_domains, losses != 0)
        norm = jnp.maximum(norm, 1.0)  # don't nan if there are no losses for a domain
        return per_domain_losses / norm

    def _domain_weighted_loss(losses, domains, alpha):
        per_domain_losses = _compute_per_domain_losses(losses, domains)  # (D,)
        return jnp.dot(alpha, per_domain_losses)  # (D,) dot (D,) -> scalar

    @partial(
        jax.jit,
        in_shardings=(rep_sharding, rep_sharding, dp_sharding),
        out_shardings=(rep_sharding, rep_sharding),
    )
    def doremi_step(state, ref_params, batch):
        rng, proxy_key, ref_key = jax.random.split(state.rng, 3)

        proxy_losses, proxy_loss_bwd = jax.vjp(
            lambda params: state.apply_fn(params, batch, rngs=dict(dropout=proxy_key)), state.params
        )
        ref_losses = ref_model.apply(
            ref_params, batch, rngs=dict(dropout=ref_key), reduce=False, train=False, method=ref_model.loss
        )

        excess_losses = proxy_losses - ref_losses
        clipped_losses = jnp.clip(excess_losses, a_min=0)

        # Change loss clipping position.
        per_domain_losses = _compute_per_domain_losses(clipped_losses, batch[domain_key])
        # per_domain_losses = jnp.clip(per_domain_losses, a_min=0, a_max=6)  # Clip above

        alpha = state.alpha * jnp.exp(domain_weight_step_size * per_domain_losses)
        alpha /= jnp.sum(alpha)
        alpha = (1 - smoothing) * alpha + initial_alpha * smoothing

        # Note DoReMi says to use the unclipped excess loss here. Confirmed with Michael
        loss, grad_loss = jax.value_and_grad(_domain_weighted_loss)(excess_losses, batch[domain_key], alpha)
        grads = proxy_loss_bwd(grad_loss)[0]

        average_alpha = state.average_alpha + (alpha - state.average_alpha) / (state.step + 1)
        new_state = state.apply_gradients(grads=grads, rng=rng, alpha=alpha, average_alpha=average_alpha)

        return new_state, dict(
            loss=loss,
            excess_loss=_compute_per_domain_losses(excess_losses, batch[domain_key]),
            ref_loss=_compute_per_domain_losses(ref_losses, batch[domain_key]),
            proxy_loss=_compute_per_domain_losses(proxy_losses, batch[domain_key]),
            lr=lr_schedule(new_state.step),
        )

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

        # Save the dataset_ids
        config_path = tf.io.gfile.join(save_path, "dataset_ids.json")
        with tf.io.gfile.GFile(config_path, "w") as f:
            json.dump(dataset_ids, f, indent=4)

        # Init wandb logging
        wandb_conf = FLAGS.config.to_dict()
        wandb_conf["checkpoint_path"] = FLAGS.checkpoint_path
        wandb_conf["checkpoint_step"] = FLAGS.checkpoint_step
        wandb.init(
            config=wandb_conf,
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
            state, info = doremi_step(state, ref_params, batch)
            train_metrics["doremi_loss"].append(info["loss"])

        step = i + 1
        if step % FLAGS.config.log_freq == 0:
            # Log training loss and timing
            # Log a BUNCH of metrics. We do this only at the log step to save time, since this has a lot of for loops.
            # For less detailed info (faster), move the following lines to right after doremi_step
            train_metrics["lr"].append(info["lr"])
            if domain_key == "dataset_id":
                for k, v in dataset_ids.items():
                    train_metrics["alpha_" + k].append(state.alpha[dataset_ids[k]])
                    train_metrics["excess_loss_" + k].append(info["excess_loss"][v])
                    train_metrics["ref_loss_" + k].append(info["ref_loss"][v])
                    train_metrics["proxy_loss_" + k].append(info["proxy_loss"][v])
            else:
                for idx in range(FLAGS.config.num_domains):
                    train_metrics["alpha_" + str(idx)].append(state.alpha[idx])
                    train_metrics["excess_loss_" + str(idx)].append(info["excess_loss"][idx])
                    train_metrics["ref_loss_" + str(idx)].append(info["ref_loss"][idx])
                    train_metrics["proxy_loss_" + str(idx)].append(info["proxy_loss"][idx])

            logger.update(train_metrics, prefix="train")
            logger.update(timer.times, prefix="time")
            logger.dump(step=step, eval=False)
            train_metrics = defaultdict(list)
            timer.reset()

        if step % FLAGS.config.save_freq == 0 and not FLAGS.debug:
            # save the train state.
            with timer("save"):
                state_checkpointer.save(
                    step, state, save_kwargs=dict(save_args=orbax_utils.save_args_from_target(state))
                )
                weights_checkpointer.save(
                    step, state.params, save_kwargs=dict(save_args=orbax_utils.save_args_from_target(state.params))
                )
            # Save the values of alpha at each step as well.
            alpha_path = tf.io.gfile.join(save_path, "alpha_{}.json".format(step))
            with tf.io.gfile.GFile(alpha_path, "w") as f:
                num_domains = len(dataset_ids) if FLAGS.config.domain_key == "dataset_id" else FLAGS.config.num_domains
                json.dump(
                    dict(
                        alpha={idx: float(state.alpha[idx]) for idx in range(num_domains)},
                        average_alpha={idx: float(state.average_alpha[idx]) for idx in range(num_domains)},
                    ),
                    f,
                    indent=4,
                )


if __name__ == "__main__":
    app.run(main)
