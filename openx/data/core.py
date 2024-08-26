import json
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from .transforms import chunk, concatenate, normalize, uniform_goal_relabeling
from .utils import DataType, NormalizationType, StateEncoding

"""
All Datasets are expected to follow the TFDS Format.
Then, we add some specific metadata, specifically ep_idx and step_idx.

This file contains functions that operate directly on dataset objects.
For general functions that should be mapped on to a dataset with dataset.map()
see transforms.py

Afterwards, we will pad the dataset to match the same thing:
{
    "observation": {
        "state": {
            StateEncoding.EE_POS:
            StateEncoding.EE_ROT6D:
        },
        "image": {
            "agent": [Img_0, Img_1], # Will randomly select one of these two for each episode.
            "wrist: [Img_0],
        },
    }
    "action": {
        "achieved_delta": [],
        "achieved_absolute": [StateEncoding.EE_POS, StateEncoding.EE_ROT6D],
        "desired_delta": [],
        "desired_absolute": [StateEncoding.GRIPPER],
    }
    "language_instruction" : "Instruction",
    "is_first": np.ndarray,
    "is_last": np.ndarray,
    "ep_idx": np.ndarray,
    "step_idx": np.ndarray,
    "ep_len": np.ndarray,
    "dataset_id": np.ndarray,
    "robot_id":
    "controller_hz":
}
"""

STANDARD_STRUCTURE = {
    "observation": {
        "state": {StateEncoding.EE_POS: NormalizationType.NONE, StateEncoding.EE_EULER: NormalizationType.NONE},
        "image": {"agent": np.zeros((224, 224), dtype=np.uint8), "wrist": np.zeros((256, 256), dtype=np.uint8)},
    },
    "action": {
        "desired_delta": {
            StateEncoding.EE_POS: NormalizationType.GAUSSIAN,
            StateEncoding.EE_ROT6D: NormalizationType.GAUSSIAN,
        },
        "desired_absolute": {StateEncoding.GRIPPER: NormalizationType.NONE},
    },
    "is_last": DataType.BOOL,
}


def _check_standard_format(dataset: tf.data.Dataset):
    element_spec = dataset.element_spec
    # TODO: implement more checks
    assert "observation" in element_spec
    assert "action" in element_spec


def filter_by_structure(tree, structure):
    if isinstance(structure, dict):
        return {k: filter_by_structure(tree[k], v) for k, v in structure.items()}
    else:
        return tree  # otherwise return the item from the tree (episode)


def filter_dataset_statistics_by_structure(dataset_statistics, structure):
    # Compute the final sa_structure we will use to filter the dataset_statistics
    # (which are computed on everything for efficiency)
    sa_structure = dict(action=structure["action"])
    if "state" in structure["observation"]:
        sa_structure["state"] = structure["observation"]["state"]
    # Filter each of the stat nested structures.
    for k in ["mean", "std", "min", "max"]:
        dataset_statistics[k] = filter_by_structure(dataset_statistics[k], sa_structure)
    return dataset_statistics


def _standardize_structure(ep, structure):
    assert "language_instruction" not in structure, "Language currently not supported."
    ep = filter_by_structure(ep, structure)  # Filter down to the keys in structure

    # Randomly subsample images.
    if "observation" in structure and "image" in structure["observation"]:
        multi_image_keys = []
        for k in structure["observation"]["image"]:
            if isinstance(ep["observation"]["image"][k], list):
                multi_image_keys.append(k)
        # Go back and edit these keys to be random
        for k in multi_image_keys:
            imgs = ep["observation"]["image"][k]
            ep["observation"]["image"][k] = imgs[tf.random.uniform((), minval=0, maxval=len(imgs))]

    return ep


def _add_metadata(ep_idx: tf.Tensor, ep: Dict[str, Any]):
    if "steps" in ep:
        steps = ep.pop("steps")
    else:
        steps = ep
    assert "is_first" in steps
    assert "is_last" in steps

    if "ep_len" not in steps:
        ep_len = tf.shape(steps["is_first"])[0]
        steps["ep_len"] = tf.repeat(ep_len, ep_len)
    else:
        ep_len = steps["ep_len"][0]
    if "ep_idx" not in steps:
        steps["ep_idx"] = tf.repeat(ep_idx, ep_len)
    if "step_idx" not in steps:
        steps["step_idx"] = tf.range(ep_len, dtype=tf.int64)

    # Could optionally broadcast the episode metadata, but for now we don't need it.
    if "episode_metadata" in ep:
        metadata = tf.nest.map_structure(lambda x: tf.repeat(x, ep_len), ep["episode_metadata"])
        steps["episode_metadata"] = metadata
    return steps


def load_dataset(
    path: str,
    split: str,
    standardization_transform: Callable,
    num_parallel_reads: Optional[int] = tf.data.AUTOTUNE,
    num_parallel_calls: Optional[int] = tf.data.AUTOTUNE,
    shuffle: bool = True,
    filter_fn: Optional[Callable] = None,  # filter functions operate on the RAW dataset pre standardization
    minimum_length: int = 3,
):
    builder = tfds.builder_from_directory(builder_dir=path)
    dataset = builder.as_dataset(
        split=split,
        decoders=dict(steps=tfds.decode.SkipDecoding()),
        shuffle_files=shuffle,
        read_config=tfds.ReadConfig(
            skip_prefetch=True,
            num_parallel_calls_for_interleave_files=num_parallel_reads,
            interleave_cycle_length=num_parallel_reads,
        ),
    )
    options = tf.data.Options()
    options.autotune.enabled = True
    options.deterministic = not shuffle
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.map_and_filter_fusion = True
    options.experimental_optimization.inject_prefetch = False
    options.experimental_warm_start = True
    dataset = dataset.with_options(options)
    # filter the dataset according to the filter function BEFORE we do anything else.
    if filter_fn is not None:
        dataset = dataset.filter(filter_fn)
    dataset = dataset.enumerate().map(_add_metadata)  # Do not parallelize the metadata call
    dataset = dataset.map(standardization_transform, num_parallel_calls=num_parallel_calls, deterministic=not shuffle)
    dataset = dataset.filter(lambda ep: ep["ep_len"][0] >= minimum_length)  # Use metadata to filter length
    _check_standard_format(dataset)
    return dataset


def compute_dataset_statistics(
    path, standardization_transform: Callable, recompute_statistics: bool = False, save_statistics: bool = True
):
    # See if we need to compute the dataset statistics
    dataset_statistics_path = tf.io.gfile.join(path, "dataset_statistics.json")
    if not recompute_statistics and tf.io.gfile.exists(dataset_statistics_path):
        with tf.io.gfile.GFile(dataset_statistics_path, "r") as f:
            dataset_statistics = json.load(f)

        # Convert everything to numpy
        def _convert_to_numpy(x):
            return {
                k: _convert_to_numpy(v) if isinstance(v, dict) else np.array(v, dtype=np.float32) for k, v in x.items()
            }

        dataset_statistics = _convert_to_numpy(dataset_statistics)
    else:
        # Otherwise, load the dataset to compute the statistics, let tf data handle the parallelization
        dataset = load_dataset(path, split="all", standardization_transform=standardization_transform)
        sa_elem_spec = dict(action=dataset.element_spec["action"], state=dataset.element_spec["observation"]["state"])
        initial_state = dict(
            num_steps=0,
            num_ep=0,
            mean=tf.nest.map_structure(lambda x: tf.zeros(x.shape[1:], dtype=np.float32), sa_elem_spec),
            var=tf.nest.map_structure(lambda x: 1e-5 * tf.ones(x.shape[1:], dtype=np.float32), sa_elem_spec),
            min=tf.nest.map_structure(lambda x: 1e10 * tf.ones(x.shape[1:], dtype=np.float32), sa_elem_spec),
            max=tf.nest.map_structure(lambda x: -1e10 * tf.ones(x.shape[1:], dtype=np.float32), sa_elem_spec),
        )

        def _reduce_fn(old_state, ep):
            # This uses a streaming algorithm to efficiently compute dataset statistics
            # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
            ep = dict(action=ep["action"], state=ep["observation"]["state"])
            tf.nest.assert_same_structure(old_state["mean"], ep)  # Check we can flatten before doing

            batch_count = tf.shape(tf.nest.flatten(ep)[0])[0] - 1  # Reduce by 1 for last transition
            batch_mean = tf.nest.map_structure(lambda x: tf.reduce_mean(x[:-1], axis=0), ep)
            batch_var = tf.nest.map_structure(lambda x: tf.math.reduce_variance(x[:-1], axis=0), ep)

            count_f, batch_count_f = (
                tf.cast(old_state["num_steps"], dtype=np.float32),
                tf.cast(batch_count, dtype=np.float32),
            )
            total_count_f = count_f + batch_count_f

            delta = tf.nest.map_structure(lambda m, m_b: m_b - m, old_state["mean"], batch_mean)
            new_mean = tf.nest.map_structure(
                lambda mean, delta: mean + delta * batch_count_f / total_count_f, old_state["mean"], delta
            )
            new_m2 = tf.nest.map_structure(
                lambda var, var_b, d: var * count_f
                + var_b * batch_count_f
                + tf.square(d) * count_f * batch_count_f / total_count_f,
                old_state["var"],
                batch_var,
                delta,
            )
            new_var = tf.nest.map_structure(lambda m2: m2 / total_count_f, new_m2)

            # Return updated values
            return dict(
                num_steps=old_state["num_steps"] + batch_count,
                num_ep=old_state["num_ep"] + 1,
                mean=new_mean,
                var=new_var,
                min=tf.nest.map_structure(
                    lambda x, m: tf.minimum(tf.reduce_min(x[:-1], axis=0), m), ep, old_state["min"]
                ),
                max=tf.nest.map_structure(
                    lambda x, m: tf.maximum(tf.reduce_max(x[:-1], axis=0), m), ep, old_state["max"]
                ),
            )

        dataset_statistics = dataset.reduce(initial_state, _reduce_fn)
        dataset_statistics["std"] = tf.nest.map_structure(lambda x: tf.math.sqrt(x), dataset_statistics["var"])
        del dataset_statistics["var"]
        dataset_statistics = tf.nest.map_structure(lambda x: x.numpy(), dataset_statistics)

        # Now re-organize the dataset statistics to be the following:
        # dict(num_ep, num_steps, state=dict(), action=di)
        dataset_statistics["num_ep"] = int(dataset_statistics["num_ep"])
        dataset_statistics["num_steps"] = int(dataset_statistics["num_steps"])

        # Save the dataset statistics
        if save_statistics:
            list_dset_stats = tf.nest.map_structure(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else x, dataset_statistics
            )
            with tf.io.gfile.GFile(dataset_statistics_path, "w") as f:
                json.dump(list_dset_stats, f, default=float, indent=4)

    return dataset_statistics


def standardize_dataset(
    dataset,
    structure,
    dataset_statistics: Optional[Dict],
    goal_conditioned: bool = False,
    n_obs: int = 1,
    n_action: int = 1,
    chunk_img: bool = True,
    shuffle: bool = True,
    repeat: bool = True,
    transforms: Optional[List] = None,
    num_parallel_calls: int = tf.data.AUTOTUNE,
):
    """
    This function standardizes the data, normalizes, concatenates, goal-conditions, chunks, and then applies transforms.
    """
    # First map away the unnecesary keys -- this reduces memory usage.
    dataset = dataset.map(
        partial(_standardize_structure, structure=structure),
        num_parallel_calls=num_parallel_calls,
        deterministic=not shuffle,
    )

    if dataset_statistics is not None:
        dataset_statistics = filter_dataset_statistics_by_structure(dataset_statistics, structure)

    # Next normalize and concatenate
    def _standardize_state_action(ep: Dict):
        if dataset_statistics is not None:
            ep = normalize(ep, structure, dataset_statistics)
        ep = concatenate(ep)
        return ep

    dataset = dataset.map(_standardize_state_action, num_parallel_calls=num_parallel_calls, deterministic=not shuffle)

    # Then, repeat the dataset. This ordering is important.
    if repeat:
        dataset = dataset.repeat()

    # Finally, chunk the dataset and apply any final transformations
    def _standardize_time(ep: Dict):
        if goal_conditioned:
            ep = uniform_goal_relabeling(ep)
        ep = chunk(ep, n_obs=n_obs, n_action=n_action, chunk_img=chunk_img)
        for transform in transforms if transforms is not None else []:
            ep = transform(ep)
        # Cut the last time step after chunking
        ep = tf.nest.map_structure(lambda x: x[:-1], ep)
        return ep

    dataset = dataset.map(_standardize_time, num_parallel_calls=num_parallel_calls, deterministic=not shuffle)

    return dataset


def flatten_dataset(dataset, num_parallel_calls: int = tf.data.AUTOTUNE, shuffle: bool = True):
    if not shuffle:
        return dataset.flat_map(tf.data.Dataset.from_tensor_slices)
    else:
        return dataset.interleave(
            lambda ep: tf.data.Dataset.from_tensor_slices(ep),
            cycle_length=num_parallel_calls,
            num_parallel_calls=num_parallel_calls,
            deterministic=not shuffle,
        )
