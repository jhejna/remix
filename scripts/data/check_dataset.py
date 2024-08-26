"""
Goes through every dataset in a config file, loads it, then displays an image and a state from the dataset.
Allows the user to check the  dataset alignment semi-manually
"""

import os
import pprint

import imageio
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from ml_collections import config_flags

from openx.data.core import load_dataset
from openx.utils.spec import ModuleSpec

FLAGS = flags.FLAGS
flags.DEFINE_string("path", "test", "Path to save the images that will be shown.")
flags.DEFINE_integer("num_data_points", 1, "Number of datapoints per dataset")
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

        builder = tfds.builder_from_directory(builder_dir=path)
        splits = builder.info.splits

        assert config["train_split"].startswith("train")

        if config.get("val_split", None) is not None and not any([config["val_split"].startswith(s) for s in splits]):
            print(
                "[INVALID SPLIT] Dataset",
                dataset_name,
                "has val split",
                config["val_split"],
                "but valid splits are only",
                splits,
            )

        dataset = load_dataset(
            path,
            "train[:" + str(FLAGS.num_data_points + 1) + "]",
            standardization_transform=transform_fn,
            shuffle=False,
        )

        # Get the first few data points
        iterator = iter(dataset)
        for i in range(FLAGS.num_data_points):
            data_pt = next(iterator)
            step = np.random.randint(0, data_pt["is_first"].shape[0] - 1)
            print("Step", data_pt["ep_idx"][step]._numpy(), data_pt["step_idx"][step]._numpy())
            image = data_pt["observation"]["image"]["agent"][step]
            assert tf.io.is_jpeg(image)
            image = tf.io.decode_jpeg(image)._numpy()
            image_path = os.path.join(FLAGS.path, dataset_name, "img_" + str(i) + ".png")
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            imageio.imwrite(image_path, image)

            if "wrist" in data_pt["observation"]["image"]:
                image = data_pt["observation"]["image"]["wrist"][step]
                assert tf.io.is_jpeg(image)
                image = tf.io.decode_jpeg(image)._numpy()
                image_path = os.path.join(FLAGS.path, dataset_name, "wrist_" + str(i) + ".png")
                imageio.imwrite(image_path, image)

            if "state" in data_pt["observation"]:
                state = tf.nest.map_structure(lambda x, step=step: x[step]._numpy(), data_pt["observation"]["state"])
                print(dataset_name, "state:")
                pprint.pprint(state)

            action = tf.nest.map_structure(lambda x, step=step: x[step]._numpy(), data_pt["action"])
            print(dataset_name, "action:")
            pprint.pprint(action)

        input("Ready to move on? [Enter]")

        # Explicit clear
        del data_pt
        del iterator
        del builder
        del dataset


if __name__ == "__main__":
    app.run(main)
