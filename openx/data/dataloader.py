import functools
from typing import Dict, List, Optional

import tensorflow as tf
import tensorflow_datasets as tfds

from openx.utils.spec import ModuleSpec

from .core import STANDARD_STRUCTURE, compute_dataset_statistics, flatten_dataset, load_dataset, standardize_dataset
from .transforms import add_dataset_id, decode_and_augment

VAL_PARALLEL_CALLS = 1


def make_dataloader(
    datasets: Dict,
    structure: Dict = STANDARD_STRUCTURE,
    dataset_statistics: Optional[Dict] = None,
    n_obs: int = 1,
    n_action: int = 1,
    chunk_img: bool = True,
    goal_conditioned: bool = False,
    global_normalization: bool = False,
    weight_by_size: bool = False,
    augment_kwargs: Optional[Dict] = None,
    batch_size: int = 256,
    shuffle_size: int = 10000,
    recompute_statistics: bool = False,
    save_statistics: bool = True,
    num_parallel_reads: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
    num_batch_parallel_calls: Optional[int] = tf.data.AUTOTUNE,
    split_for_jax: bool = True,
    episode_transforms: Optional[List] = None,
    image_transforms: Optional[List] = None,
    ram_budget: Optional[float] = None,
    restrict_memory: bool = False,
):
    # Get all datasets
    train_datasets = dict()
    val_datasets = dict()
    weights = dict()
    if dataset_statistics is None:
        dataset_statistics = dict()

    # Loop through all datasets to construct dataloader
    for dataset, config in datasets.items():
        assert "path" in config and "transform" in config
        transform_fn = ModuleSpec.instantiate(config["transform"])
        filter_fn = ModuleSpec.instantiate(config["filter"]) if config.get("filter", None) is not None else None
        path = config["path"]
        reads = config.get("num_parallel_reads", num_parallel_reads)
        calls = config.get("num_parallel_calls", num_parallel_calls)
        stats_path = config.get("dataset_statistics", path)

        # Compute dataset statistics
        if dataset not in dataset_statistics:
            dataset_statistics[dataset] = compute_dataset_statistics(
                stats_path, transform_fn, recompute_statistics=recompute_statistics, save_statistics=save_statistics
            )

        # Add train split
        if "train_split" in config and config["train_split"] is not None:
            train_datasets[dataset] = load_dataset(
                path,
                tfds.split_for_jax_process(config["train_split"]) if split_for_jax else config["train_split"],
                standardization_transform=transform_fn,
                filter_fn=filter_fn,
                num_parallel_reads=reads,
                num_parallel_calls=calls,
                shuffle=shuffle_size > 0,
            )
            weight = config.get("weight", 1.0)
            if weight_by_size and filter_fn is not None:
                # If we have filtered the dataset we need to determine the actual size
                num_ep, num_steps = train_datasets[dataset].reduce(
                    (0, 0), lambda state, ep: (state[0] + 1, state[1] + ep["ep_len"][0] - 1)
                )
                dataset_statistics[dataset]["num_ep"] = int(num_ep.numpy())
                dataset_statistics[dataset]["num_steps"] = int(num_steps.numpy())

            if weight_by_size:
                # This is not exact since there is a train / test split, but avoids a whole dataset iteration.
                weight *= dataset_statistics[dataset]["num_steps"]

            weights[dataset] = weight

        # Add val split if present
        if "val_split" in config and config["val_split"] is not None:
            # Val dataloading is the same except we limit the number of workers
            val_datasets[dataset] = load_dataset(
                path,
                config["val_split"],  # Do not split validation set.
                standardization_transform=transform_fn,
                filter_fn=filter_fn,
                num_parallel_reads=VAL_PARALLEL_CALLS,
                num_parallel_calls=VAL_PARALLEL_CALLS,
                shuffle=shuffle_size > 0,
            )
            # No weights are used for validation datasets.

    # Next, determine how we are going to normalize the datasets
    if global_normalization:
        # Can apply a global normalization to all datasets
        # for now leave this unimplemented
        raise NotImplementedError

    # Next standardize all the datasets. This is done after creation if we want global normalization.
    _standardize = functools.partial(
        standardize_dataset,
        structure=structure,
        n_obs=n_obs,
        n_action=n_action,
        goal_conditioned=goal_conditioned,
        chunk_img=chunk_img,
        shuffle=shuffle_size > 0,
        transforms=episode_transforms,
    )
    train_datasets = {
        k: _standardize(
            v,
            dataset_statistics=dataset_statistics[k],
            num_parallel_calls=datasets[k].get("num_parallel_calls", num_parallel_calls),
        )
        for k, v in train_datasets.items()
    }
    val_datasets = {
        k: _standardize(v, dataset_statistics=dataset_statistics[k], num_parallel_calls=VAL_PARALLEL_CALLS)
        for k, v in val_datasets.items()
    }

    # Add dataset IDs to every dataset
    # NOTE: Perhaps we should move this into standardize to be faster.
    dataset_ids = sorted(list(set(list(train_datasets.keys()) + list(val_datasets.keys()))))
    dataset_ids = {k: v for v, k in enumerate(dataset_ids)}
    train_datasets = {
        k: v.map(
            functools.partial(add_dataset_id, dataset_id=dataset_ids[k]),
            num_parallel_calls=datasets[k].get("num_parallel_calls", num_parallel_calls),
            deterministic=shuffle_size > 0,
        )
        for k, v in train_datasets.items()
    }
    val_datasets = {
        k: v.map(
            functools.partial(add_dataset_id, dataset_id=dataset_ids[k]),
            num_parallel_calls=VAL_PARALLEL_CALLS,
            deterministic=shuffle_size > 0,
        )
        for k, v in val_datasets.items()
    }

    # Then flatten the datasets
    train_datasets = {
        k: flatten_dataset(
            v, shuffle=shuffle_size > 0, num_parallel_calls=datasets[k].get("num_parallel_calls", num_parallel_calls)
        )
        for k, v in train_datasets.items()
    }
    val_datasets = {
        k: flatten_dataset(v, shuffle=shuffle_size > 0, num_parallel_calls=VAL_PARALLEL_CALLS)
        for k, v in val_datasets.items()
    }

    # Combine the train datasets into one dataset
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    print("\n######################################################################################")
    print(f"# Loading the following {len(train_datasets)} datasets (incl. sampling weight):{'': >24} #")
    for dataset_name in train_datasets.keys():
        pad = 80 - len(dataset_name)
        print(f"# {dataset_name}: {weights[dataset_name]:=>{pad}f} #")
    print("######################################################################################\n")

    if len(train_datasets) > 1:
        order = sorted(list(train_datasets.keys()))
        train_dataset = tf.data.Dataset.sample_from_datasets(
            [train_datasets[p] for p in order], weights=[weights[p] for p in order]
        )
    else:
        train_dataset = train_datasets[next(iter(train_datasets.keys()))]

    # Shuffle the datasets
    if shuffle_size > 0:
        train_dataset = train_dataset.shuffle(shuffle_size)
        val_datasets = {
            k: v.shuffle(
                int(shuffle_size * weights.get(k, 1 / len(val_datasets)) // 10),
            )
            for k, v in val_datasets.items()
        }

    # Decode and augment the images
    augment_kwargs = dict() if augment_kwargs is None else augment_kwargs
    _decode_and_augment = functools.partial(decode_and_augment, structure=structure, **augment_kwargs)
    train_dataset = train_dataset.map(
        functools.partial(_decode_and_augment, train=True),
        num_parallel_calls=num_parallel_calls,
        deterministic=shuffle_size > 0,
    )
    val_datasets = {
        k: v.map(
            functools.partial(_decode_and_augment, train=False),
            num_parallel_calls=4 * VAL_PARALLEL_CALLS,
            deterministic=shuffle_size > 0,
        )
        for k, v in val_datasets.items()
    }

    # Apply more image transforms, if we have any. This is usually empty.
    for image_transform in image_transforms if image_transforms is not None else []:
        transform_fn = ModuleSpec.instantiate(image_transform)
        train_dataset = train_dataset.map(
            functools.partial(transform_fn, train=True),
            num_parallel_calls=num_parallel_calls,
            deterministic=shuffle_size > 0,
        )
        val_datasets = {
            k: v.map(
                functools.partial(transform_fn, train=False),
                num_parallel_calls=4 * VAL_PARALLEL_CALLS,
                deterministic=shuffle_size > 0,
            )
            for k, v in val_datasets.items()
        }

    # Finally, batch the datasets
    train_dataset = train_dataset.batch(
        batch_size, num_parallel_calls=num_batch_parallel_calls, drop_remainder=True
    )  # num_parallel_calls if parallel_batch else None
    val_num_batch_parallel_calls = VAL_PARALLEL_CALLS if num_batch_parallel_calls is not None else None
    val_datasets = {
        k: v.batch(batch_size, num_parallel_calls=val_num_batch_parallel_calls, drop_remainder=True)
        for k, v in val_datasets.items()
    }

    # Then, add memory limits for autotune.
    if restrict_memory:
        train_options = tf.data.Options()
        train_options.autotune.ram_budget = int(4 * 1024 * 1024 * 1024)  # GB -> Bytes
        train_dataset = train_dataset.with_options(train_options)
        val_options = tf.data.Options()
        val_options.autotune.ram_budget = int(1 * 1024 * 1024 * 1024)  # GB -> Bytes
        val_datasets = {k: v.with_options(val_options) for k, v in val_datasets.items()}

    return train_dataset, val_datasets, dataset_statistics, dataset_ids
