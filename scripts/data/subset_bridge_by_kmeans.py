import functools
import os
from typing import Dict

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string("kmeans_path", "1.0.0_kmeans.npy", "Path to the kmeans file.")
flags.DEFINE_string("dataset_path", None, "Path to tfds dataset")
flags.DEFINE_float("percent", 10, "Percent of dataset to subset")
flags.DEFINE_string("data_dir", None, "Path to save subsets")


def assign(ep: Dict, centroids):
    embedding = ep["episode_metadata"]["embedding"]
    distances = tf.reduce_sum((embedding[None, :] - centroids) ** 2, axis=-1)  # (K,)
    centroid = tf.argmin(distances, axis=-1)
    return centroid, distances[centroid]


def main(_):
    with open(FLAGS.kmeans_path, "rb") as f:
        centroids = np.load(f)

    builder = tfds.builder_from_directory(builder_dir=FLAGS.dataset_path)

    # First determine the thresholds for every cluster
    dataset = builder.as_dataset(
        split="train",
        decoders=dict(steps=tfds.decode.SkipDecoding()),
        read_config=tfds.ReadConfig(
            skip_prefetch=True,
            num_parallel_calls_for_interleave_files=8,
            interleave_cycle_length=8,
        ),
    )
    dataset = dataset.map(functools.partial(assign, centroids=centroids), num_parallel_calls=64, deterministic=False)

    assignments, distances = [], []
    for assignment, distance in tqdm.tqdm(dataset.as_numpy_iterator(), dynamic_ncols=True):
        assignments.append(assignment)
        distances.append(distance)
    assignments = np.array(assignments)
    distances = np.array(distances)

    # For each centroid compute the minimum distance
    minimum_distances = dict()
    for i in range(centroids.shape[0]):
        cur_dists = distances[assignments == i]
        minimum_distances[i] = np.quantile(cur_dists, 1 - (FLAGS.percent / 100))

    def val_generator(split):
        # Just a pass through that returns all the original data
        dataset = builder.as_dataset(
            split=split,
            decoders=dict(
                steps=tfds.decode.SkipDecoding(),
            ),
            read_config=tfds.ReadConfig(
                skip_prefetch=True,
                num_parallel_calls_for_interleave_files=8,
                interleave_cycle_length=8,
            ),
        )
        for ep in dataset.as_numpy_iterator():
            yield ep["episode_metadata"]["file_path"], ep

    def train_generator(split):
        dataset = builder.as_dataset(
            split=split,
            decoders=dict(steps=tfds.decode.SkipDecoding()),
            read_config=tfds.ReadConfig(
                skip_prefetch=True,
                num_parallel_calls_for_interleave_files=8,
                interleave_cycle_length=8,
            ),
        )

        size = dataset.cardinality().numpy()
        size = int(size * FLAGS.percent / 100)
        num = 0

        for ep in dataset.as_numpy_iterator():
            # compute the datapoint for the centroid
            embedding = ep["episode_metadata"]["embedding"]
            distances = ((embedding[None, :] - centroids) ** 2).sum(axis=-1)  # (K,)
            assignment = np.argmin(distances)
            distance = distances[assignment]

            if distance >= minimum_distances[assignment]:
                num += 1
                yield ep["episode_metadata"]["file_path"], ep

                if num == size:
                    return

    # Create the directory
    name = "bridge_kmeans_" + str(FLAGS.percent)
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
