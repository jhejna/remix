import os
from functools import partial
from typing import Dict

import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from jax import numpy as jnp

"""
Script that runs K-means on a dataset and saves the result.
Taken from: https://colab.research.google.com/drive/1AwS4haUx6swF82w3nXr6QKhajdF8aSvA
"""

FLAGS = flags.FLAGS

flags.DEFINE_string("dataset_path", None, "Path to tfds dataset with embeddings")
flags.DEFINE_integer("data_points", 10000, "number of data points to use")
flags.DEFINE_integer("k", 32, "k in k-means")


@jax.jit
def vector_quantize(points, codebook):
    assignment = jax.vmap(lambda point: jnp.argmin(jax.vmap(jnp.linalg.norm)(codebook - point)))(points)
    distns = jax.vmap(jnp.linalg.norm)(codebook[assignment, :] - points)
    return assignment, distns


@partial(jax.jit, static_argnums=(2,))
def kmeans_run(key, points, k, thresh=1e-4):
    def improve_centroids(val):
        prev_centroids, prev_distn, _ = val
        assignment, distortions = vector_quantize(points, prev_centroids)

        # Count number of points assigned per centroid
        # (Thanks to Jean-Baptiste Cordonnier for pointing this way out that is
        # much faster and let's be honest more readable!)
        counts = (
            (assignment[jnp.newaxis, :] == jnp.arange(k)[:, jnp.newaxis])
            .sum(axis=1, keepdims=True)
            .clip(min=1.0)  # clip to change 0/0 later to 0/1
        )

        # Sum over points in a centroid by zeroing others out
        new_centroids = (
            jnp.sum(
                jnp.where(
                    # axes: (data points, clusters, data dimension)
                    assignment[:, jnp.newaxis, jnp.newaxis] == jnp.arange(k)[jnp.newaxis, :, jnp.newaxis],
                    points[:, jnp.newaxis, :],
                    0.0,
                ),
                axis=0,
            )
            / counts
        )

        return new_centroids, jnp.mean(distortions), prev_distn

    # Run one iteration to initialize distortions and cause it'll never hurt...
    initial_indices = jax.random.shuffle(key, jnp.arange(points.shape[0]))[:k]
    initial_val = improve_centroids((points[initial_indices, :], jnp.inf, None))
    # ...then iterate until convergence!
    centroids, distortion, _ = jax.lax.while_loop(
        lambda val: (val[2] - val[1]) > thresh,
        improve_centroids,
        initial_val,
    )
    return centroids, distortion


def get_embedding(ep: Dict):
    return ep["episode_metadata"]["embedding"]


def main(_):
    builder = tfds.builder_from_directory(builder_dir=FLAGS.dataset_path)

    dataset = builder.as_dataset(
        split="train[:" + str(FLAGS.data_points) + "]", decoders=dict(steps=tfds.decode.SkipDecoding())
    )
    dataset = dataset.take(FLAGS.data_points)
    dataset = dataset.map(lambda ep: ep["episode_metadata"]["embedding"], num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(FLAGS.data_points)
    dataset = dataset.take(1)  # just take the first item :)
    dataset = next(iter(dataset.as_numpy_iterator()))

    # Now run k-means on everything.
    key = jax.random.PRNGKey(0)
    centroids, distortions = kmeans_run(key, dataset, FLAGS.k)

    dataset_name = os.path.basename(os.path.dirname(FLAGS.dataset_path))
    with open(dataset_name + "_kmeans.npy", "wb") as f:
        np.save(f, np.asarray(centroids))

    # Save the centroids to a json file so we can re-use them later.


if __name__ == "__main__":
    app.run(main)
