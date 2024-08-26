from functools import partial
from typing import Dict, Optional, Tuple

import tensorflow as tf

from .utils import NormalizationType

OBSERVATION_KEYS = ("observation", "goal")


def _observation_transform(transform):
    """
    Takes a transform designed to work on observations and applies it to all potential observation
    types, including: "observation", "goal", "next_observation" (not implemented yet)
    """

    def fn(ep, *args, **kwargs):
        for k in OBSERVATION_KEYS:
            if k in ep:
                ep[k] = transform(ep[k], *args, **kwargs)
        return ep

    return fn


def chunk(ep: Dict, n_obs: int, n_action: int, chunk_img: bool = True, mask_keys=("state",)):
    """
    Chunk an episode into observation and action pairs.
    Sequences are done as:
    (obs_{t-n_obs + 1}, ... obs_t)
    (a_t, a_{t+1}, a_{t+n_act - 1})

    This function additionally adds a mask parameter to dataset for the actions
    We don't mask observation sequences (and instead will repeat the first observation).

    After calling this function we assume all data has a time dimension.

    Code inspired from:
    https://github.com/jhejna/openx-lightning/blob/rl/openx/datasets/replay_buffer/sampling.py#L79
    """
    ep_len = tf.shape(tf.nest.flatten(ep["action"])[0])[0]
    # Observation chunking
    idx = tf.range(ep_len)
    obs_idx = tf.maximum(idx[:, None] + tf.range(-n_obs + 1, 1), 0)

    # Action chunking
    action_idx = idx[:, None] + tf.range(0, n_action)
    mask = action_idx < ep_len - 1  # The last action is also invalid!!!!
    # Handle goals
    if "goal_idx" in ep:  # If we have goals ensure its less than the goal index
        mask = action_idx < ep["goal_idx"][:, None]
        del ep["goal_idx"]
        ep["goal"] = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=1), ep["goal"])
    action_idx = tf.minimum(action_idx, ep_len - 1)  # mask the actual indexes to not go over.
    ep["mask"] = mask

    # Apply indexing
    if chunk_img:
        ep["observation"] = tf.nest.map_structure(lambda x: tf.gather(x, obs_idx), ep["observation"])
        history_mask = tf.constant([False] * (n_obs - 1) + [True])
        for mask_key in mask_keys:
            if mask_key in ep["observation"]:
                ep["observation"][mask_key] = tf.nest.map_structure(
                    lambda x: x * tf.reshape(tf.cast(history_mask, x.dtype), (1, -1) + (1,) * (x.ndim - 2)),
                    ep["observation"][mask_key],
                )
    else:
        if "state" in ep["observation"]:
            ep["observation"]["state"] = tf.gather(ep["observation"]["state"], obs_idx)
        if "image" in ep["observation"]:
            ep["observation"]["image"] = tf.nest.map_structure(
                lambda x: tf.expand_dims(x, axis=0), ep["observation"]["image"]
            )
    ep["action"] = tf.gather(ep["action"], action_idx)
    return ep


def concatenate(ep):
    """
    Concatenates all state and action keys into a fixed order
    """
    if "state" in ep["observation"]:
        ep["observation"]["state"] = tf.concat(tf.nest.flatten(ep["observation"]["state"]), axis=-1)
    ep["action"] = tf.concat(tf.nest.flatten(ep["action"]), axis=-1)
    return ep


def uniform_goal_relabeling(ep: Dict):
    ep_len = tf.shape(tf.nest.flatten(ep["observation"])[0])[0]
    rand = tf.random.uniform([ep_len])
    # We shift the low and high a bit so a goal is always a future state, cannot be a current state.
    low, high = tf.cast(tf.range(ep_len) + 1, tf.float32), tf.cast(ep_len, tf.float32) + 1e-5
    goal_idx = tf.minimum(tf.cast(rand * (high - low) + low, tf.int32), ep_len - 1)
    ep["goal"] = tf.nest.map_structure(lambda x: tf.gather(x, goal_idx), ep["observation"])
    ep["goal_idx"] = goal_idx
    return ep


def _normalize(x, mode: NormalizationType, mean, std, low, high):
    if mode == NormalizationType.NONE:
        return x
    elif mode == NormalizationType.GAUSSIAN:
        return tf.math.divide_no_nan(x - mean, std)
    elif mode == NormalizationType.BOUNDS:
        x = tf.clip_by_value(x, low, high)
        # Apply divide_no_nan to allow for constant fields.
        return 2 * tf.math.divide_no_nan(x - low, high - low) - 1
    elif mode == NormalizationType.BOUNDS_5STDV:
        low = tf.maximum(low, mean - 5 * std)
        high = tf.minimum(high, mean + 5 * std)
        x = tf.clip_by_value(x, low, high)
        # Apply divide_no_nan to allow for constant fields.
        return 2 * tf.math.divide_no_nan(x - low, high - low) - 1
    else:
        raise ValueError("Invalid Mode selected")


def _unnormalize(x, mode: NormalizationType, mean, std, low, high):
    if mode == NormalizationType.NONE:
        return x
    elif mode == NormalizationType.GAUSSIAN:
        return std * x + mean
    elif mode == NormalizationType.BOUNDS:
        x = tf.clip_by_value(x, -1, 1)
        return ((x + 1) / 2) * (high - low) + low
    elif mode == NormalizationType.BOUNDS_5STDV:
        low = tf.maximum(low, mean - 5 * std)
        high = tf.minimum(high, mean + 5 * std)
        x = tf.clip_by_value(x, -1, 1)
        return ((x + 1) / 2) * (high - low) + low
    else:
        raise ValueError("Invalid Mode selected")


def normalize(ep, structure, dataset_statistics):
    if "state" in ep["observation"]:
        tf.nest.assert_same_structure(dataset_statistics["mean"]["state"], ep["observation"]["state"])
        ep["observation"]["state"] = tf.nest.map_structure(
            _normalize,
            ep["observation"]["state"],
            structure["observation"]["state"],
            dataset_statistics["mean"]["state"],
            dataset_statistics["std"]["state"],
            dataset_statistics["min"]["state"],
            dataset_statistics["max"]["state"],
        )
    tf.nest.assert_same_structure(dataset_statistics["mean"]["action"], ep["action"])
    ep["action"] = tf.nest.map_structure(
        _normalize,
        ep["action"],
        structure["action"],
        dataset_statistics["mean"]["action"],
        dataset_statistics["std"]["action"],
        dataset_statistics["min"]["action"],
        dataset_statistics["max"]["action"],
    )
    return ep


def _sample_random_bbox(
    img_shape,
    desired_shape,
    scale_range: Optional[Tuple[float, float]] = None,
    aspect_ratio_range: Optional[Tuple[float, float]] = None,
):
    im_h, im_w = img_shape[-3], img_shape[-2]
    im_h, im_w = tf.cast(im_h, dtype=tf.float32), tf.cast(im_w, dtype=tf.float32)
    if aspect_ratio_range is None:
        ratio = desired_shape[-1] / desired_shape[-2]  # Width / Height
    else:
        ratio = tf.random.uniform((), minval=aspect_ratio_range[0], maxval=aspect_ratio_range[1])
    if scale_range is None:
        scale = 1
    else:
        scale = tf.random.uniform((), minval=scale_range[0], maxval=scale_range[1], dtype=tf.float32)
    b_h = tf.minimum(im_h, im_w / ratio)
    b_w = b_h * ratio
    b_h, b_w = b_h * scale, b_w * scale
    o_h = tf.random.uniform((), minval=0, maxval=im_h - b_h + 1, dtype=tf.float32)  # Need to add a one offset.
    o_w = tf.random.uniform((), minval=0, maxval=im_w - b_w + 1, dtype=tf.float32)  # Need to add a one offset.
    return tf.cast(tf.stack([o_h, o_w, b_h, b_w]), dtype=tf.int32)


def _center_bbox(img_shape, desired_shape, scale_range: Optional[Tuple[float, float]] = None):
    im_h, im_w = img_shape[-3], img_shape[-2]
    im_h, im_w = tf.cast(im_h, dtype=tf.float32), tf.cast(im_w, dtype=tf.float32)
    ratio = desired_shape[-1] / desired_shape[-2]  # Width / Height
    if scale_range is None:
        scale = 1
    else:
        scale = (scale_range[1] + scale_range[0]) / 2
    b_h = tf.minimum(im_h, im_w / ratio)
    b_w = b_h * ratio
    b_h, b_w = b_h * scale, b_w * scale
    o_h, o_w = (im_h - b_h) / 2, (im_w - b_w) / 2
    return tf.cast(tf.stack([o_h, o_w, b_h, b_w]), dtype=tf.int32)


def _decode_and_augment(
    img_bytes_stack,
    bbox: tf.Tensor,
    seed: tf.Tensor,
    desired_shape: Tuple[int, int],
    brightness: Optional[float] = None,
    contrast_range: Optional[Tuple[float, float]] = None,
    saturation_range: Optional[Tuple[float, float]] = None,
    hue: Optional[float] = None,
):
    imgs = tf.stack(
        [tf.io.decode_and_crop_jpeg(image_bytes, bbox, channels=3) for image_bytes in tf.unstack(img_bytes_stack)]
    )
    imgs = tf.image.convert_image_dtype(imgs, dtype=tf.float32)
    imgs = tf.image.resize(imgs, desired_shape)
    if brightness is not None:
        seed = tf.random.stateless_uniform([2], seed, maxval=tf.dtypes.int32.max, dtype=tf.int32)
        imgs = tf.image.stateless_random_brightness(imgs, max_delta=brightness, seed=seed)
    if contrast_range is not None:
        seed = tf.random.stateless_uniform([2], seed, maxval=tf.dtypes.int32.max, dtype=tf.int32)
        imgs = tf.image.stateless_random_contrast(imgs, lower=contrast_range[0], upper=contrast_range[1], seed=seed)
    if saturation_range is not None:
        seed = tf.random.stateless_uniform([2], seed, maxval=tf.dtypes.int32.max, dtype=tf.int32)
        imgs = tf.image.stateless_random_saturation(
            imgs, lower=saturation_range[0], upper=saturation_range[1], seed=seed
        )
    if hue is not None:
        seed = tf.random.stateless_uniform([2], seed, maxval=tf.dtypes.int32.max, dtype=tf.int32)
        imgs = tf.image.stateless_random_hue(imgs, max_delta=hue, seed=seed)
    # Images are expected to be in 0 to 1 after decoding and convertion to float32
    imgs = tf.clip_by_value(imgs, 0, 1)
    return imgs


def decode_and_augment(
    step: Dict,
    structure: Dict,
    scale_range: Optional[Tuple[float, float]] = None,
    aspect_ratio_range: Optional[Tuple[float, float]] = None,
    brightness: Optional[float] = None,
    contrast_range: Optional[Tuple[float, float]] = None,
    saturation_range: Optional[Tuple[float, float]] = None,
    hue: Optional[float] = None,
    aligned: bool = True,
    train: bool = False,
):
    if "image" not in structure["observation"]:
        return step

    # Define bounding boxes
    def _get_bbox(img_stack, structure):
        img_shape = tf.io.extract_jpeg_shape(img_stack[0])
        if train:
            return _sample_random_bbox(
                img_shape, structure, scale_range=scale_range, aspect_ratio_range=aspect_ratio_range
            )
        else:
            return _center_bbox(img_shape, structure, scale_range=scale_range)

    # Define augmentations
    if train:
        _bound_decode_and_augment = partial(
            _decode_and_augment,
            brightness=brightness,
            contrast_range=contrast_range,
            saturation_range=saturation_range,
            hue=hue,
        )
    else:
        _bound_decode_and_augment = partial(
            _decode_and_augment, brightness=None, contrast_range=None, saturation_range=None, hue=None
        )

    # Here we assume that images folow the exact keys, so we don't do nesting.
    if aligned:
        bboxes = {
            k: _get_bbox(step["observation"]["image"][k], v) for k, v in structure["observation"]["image"].items()
        }
        seeds = {
            k: tf.random.uniform([2], maxval=tf.dtypes.int32.max - 3, dtype=tf.int32)
            for k in structure["observation"]["image"].keys()
        }
    else:
        bboxes, seeds = None, None

    @_observation_transform
    def _apply(observation, structure, bboxes=None, seeds=None):
        # If align is false, construct boxes and seeds to get different results for obs/goal/next_obs images.
        if bboxes is None:
            bboxes = {k: _get_bbox(observation["image"][k], v) for k, v in structure["image"].items()}
        if seeds is None:
            seeds = {
                k: tf.random.uniform([2], maxval=tf.dtypes.int32.max, dtype=tf.int32) for k in structure["image"].keys()
            }

        # Map the augmentations over the images.
        observation["image"] = {
            k: _bound_decode_and_augment(observation["image"][k], bboxes[k], seeds[k], v)
            for k, v in structure["image"].items()
        }
        return observation

    step = _apply(step, structure["observation"], bboxes=bboxes, seeds=seeds)
    return step


def add_dataset_id(ep: Dict, dataset_id: int):
    ep_len = tf.shape(tf.nest.flatten(ep["action"])[0])[0]
    ep["dataset_id"] = tf.repeat(dataset_id, ep_len)
    return ep


def random_noised_actions(step: Dict, train: bool = True, freq: int = 4):
    if len(tf.shape(step["ep_idx"])) > 0:
        ep_idx = step["ep_idx"][0]
    else:
        ep_idx = step["ep_idx"]
    predicate = tf.math.floormod(ep_idx, freq) == 0

    step["action"] = tf.cond(
        predicate,
        lambda: tf.nest.map_structure(lambda x: tf.random.normal(tf.shape(x), dtype=x.dtype), step["action"]),
        lambda: step["action"],
    )

    return step


def make_actions_zero(step: Dict, train: bool = True):
    """
    Used for debugging.
    """
    step["action"] = tf.nest.map_structure(lambda x: tf.zeros(x.shape, dtype=x.dtype), step["action"])
    return step


def blur_images(
    ep: Dict,
    dataset_id: Optional[int] = None,
    filter_size: int = 5,
    sigma: float = 1.0,
    train: bool = True,
    aligned: bool = True,
):
    """
    Blurs all images. This is used for debugging doremi.
    Based on: https://github.com/tensorflow/models/blob/v2.15.0/official/vision/ops/augment.py#L184-L270
    """
    # See if we need to perform blur
    if ep["dataset_id"] != dataset_id:
        return ep

    @_observation_transform
    def _apply(obs: Dict):
        kernel = tf.cast(tf.range(-filter_size // 2 + 1, filter_size // 2 + 1), dtype=tf.float32)
        kernel = tf.exp(-tf.pow(kernel, 2.0) / (2.0 * (sigma**2)))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.matmul(kernel[:, tf.newaxis], kernel[tf.newaxis, :])  # (K, 1) x (1 x K)
        kernel = kernel[:, :, tf.newaxis, tf.newaxis]  # (K, K, 1, 1)

        def _blur(image_stack):
            num_channels = tf.shape(image_stack)[-1]
            gaussian_kernel = tf.tile(kernel, [1, 1, num_channels, 1])
            blurred = tf.nn.depthwise_conv2d(image_stack, gaussian_kernel, strides=[1, 1, 1, 1], padding="SAME")
            return blurred

        obs["image"] = tf.nest.map_structure(_blur, obs["image"])
        return obs

    return _apply(ep)
