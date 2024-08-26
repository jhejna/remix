"""
Goes through every dataset in a config file, loads it, then displays an image and a state from the dataset.
Allows the user to check the  dataset alignment semi-manually
"""

import os

import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags
from matplotlib import pyplot as plt
from ml_collections import config_flags

from openx.data.core import compute_dataset_statistics, load_dataset, standardize_dataset
from openx.data.utils import NormalizationType
from openx.utils.spec import ModuleSpec

FLAGS = flags.FLAGS
flags.DEFINE_string("path", "action_distribtuions", "Path to save the images that will be shown.")
# Always lock the config to avoid subtle bugs
config_flags.DEFINE_config_file(
    "config", None, "File path to the training hyperparameter configuration.", lock_config=True
)


def main(_):
    # Make the directory where we will write images
    os.makedirs(FLAGS.path, exist_ok=True)

    # Iterate through all datasets
    datasets = FLAGS.config.dataloader.datasets.to_dict()
    for dataset_name, config in datasets.items():
        assert "path" in config and "transform" in config
        transform_fn = ModuleSpec.instantiate(config["transform"])
        path = config["path"]
        structure = FLAGS.config.structure.to_dict()

        dataset = load_dataset(path, "all", standardization_transform=transform_fn, shuffle=False)
        dataset_statistics = compute_dataset_statistics(
            path, transform_fn, recompute_statistics=False, save_statistics=False
        )

        dataset = standardize_dataset(dataset, structure, dataset_statistics, n_action=1, repeat=False)

        # Get the first few data points
        iterator = iter(dataset)
        all_actions = []
        dataset_size = dataset.cardinality().numpy()
        for batch in tqdm.tqdm(iterator, total=dataset_size if dataset_size != tf.data.UNKNOWN_CARDINALITY else None):
            # Get the actions
            actions = batch["action"]._numpy()  # Shape (T, 1, D)
            all_actions.append(actions[:, 0, :])

        actions = np.concatenate(all_actions, axis=0)  # (Datset Size, D)

        # Determine normalization type
        normalization_type = next(x for x in tf.nest.flatten(structure["action"]) if x != NormalizationType.NONE)

        if normalization_type == NormalizationType.BOUNDS or NormalizationType.BOUNDS_5STDV:
            hist_range = [-1, 1]
        if normalization_type == NormalizationType.GAUSSIAN:
            # Set it to 4 std dev
            hist_range = [-5, 5]

        bins = 128
        # bins = norm.ppf(np.linspace(5e-3, 1 - 5e-3, bins + 1), scale=2)

        D = actions.shape[-1]
        fig, axes = plt.subplots(D, 1, figsize=(10, 2 * D))
        for i in range(D):
            axes[i].hist(actions[:, i], bins=bins, alpha=0.75, range=hist_range, edgecolor="black")
            axes[i].set_title(
                f"{dataset_name} Dim {i+1} {normalization_type}" + ("" if isinstance(bins, int) else " PPF")
            )
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Frequency")
            axes[i].set_xlim(hist_range)
            axes[i].set_yscale("log")

        # save the plot
        plt.tight_layout(pad=1)
        filename = dataset_name + f"_{normalization_type}"
        if not isinstance(bins, int):
            filename += "_ppf"
        plt.savefig(os.path.join(FLAGS.path, filename + ".png"), dpi=300)
        plt.clf()


if __name__ == "__main__":
    app.run(main)
