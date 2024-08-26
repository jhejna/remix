import json
import os
import pprint
from functools import partial

import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags

from openx.data.datasets.bridge import FULL_BRIDGE_WEIGHTS, FULL_DOMAIN_TABLE

DOMAIN_TABLE = FULL_DOMAIN_TABLE
DOMAIN_WEIGHTS = FULL_BRIDGE_WEIGHTS

FLAGS = flags.FLAGS

flags.DEFINE_string("weights", None, "Path to the DoReMi weights", required=True)
flags.DEFINE_string("dataset_path", None, "Path to tfds dataset", required=True)
flags.DEFINE_string("data_dir", None, required=True)
flags.DEFINE_float("percent", 10, "Percent of dataset to subset")


def main(_):
    with tf.io.gfile.GFile(FLAGS.weights, "r") as f:
        domain_weights = json.load(f)["average_alpha"]  # load the average alpha value over the course of training
        domain_weights = {i: domain_weights[str(i)] for i in range(len(domain_weights))}
    print("##### DOMAIN WEIGHTS ######")
    pprint.pprint(domain_weights)

    builder = tfds.builder_from_directory(builder_dir=FLAGS.dataset_path)

    def val_generator(split):
        # Just a pass through that returns all the original data
        dataset = builder.as_dataset(
            split=split,
            decoders=dict(steps=tfds.decode.SkipDecoding()),
        )
        for ep in dataset.as_numpy_iterator():
            yield ep["episode_metadata"]["file_path"], ep

    def train_generator(split):
        dataset = builder.as_dataset(
            split=split,
            decoders=dict(steps=tfds.decode.SkipDecoding()),
        )
        size = dataset.cardinality().numpy()
        domain_sizes = {domain: int(size * DOMAIN_WEIGHTS[domain]) for domain in domain_weights.keys()}
        assert size > 1
        size = int(size * FLAGS.percent / 100)
        num_per_domain = {domain: int(round(size * weight)) for domain, weight in domain_weights.items()}
        for domain in domain_weights.keys():
            assert domain_sizes[domain] >= num_per_domain[domain], "Error -- not enough data in domain for true subset"

        for ep in dataset.as_numpy_iterator():
            # Get the datapoint domain
            parts = tf.strings.split(ep["episode_metadata"]["file_path"], "/")
            dataset_name = parts[4]

            string_key = tf.cond(
                dataset_name == "bridge_data_v2" or dataset_name == "rss",
                partial(lambda x: x, parts[5]),
                partial(lambda x: x, parts[6]),
            )
            scene_id = DOMAIN_TABLE.lookup(string_key)
            scene_id = int(scene_id.numpy())

            if num_per_domain[scene_id] > 0:
                num_per_domain[scene_id] -= 1
                yield ep["episode_metadata"]["file_path"], ep

                if all([v <= 0 for v in num_per_domain.values()]):
                    return

    # Create the directory
    name = "bridge_doremi_" + str(FLAGS.percent)
    tf.io.gfile.makedirs(os.path.join(FLAGS.data_dir, name))

    subset_dataset_builder = tfds.dataset_builders.AdhocBuilder(
        name=name,
        version=builder.version,
        features=builder.info.features,
        split_datasets=dict(train=train_generator("train"), val=val_generator("val")),
        config=builder.builder_config,
        data_dir=FLAGS.data_dir,
    )
    subset_dataset_builder.download_and_prepare()  # save the dataset


if __name__ == "__main__":
    app.run(main)
