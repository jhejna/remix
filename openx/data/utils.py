from enum import StrEnum

import tensorflow as tf


class StateEncoding(StrEnum):
    """Defines state keys for datasets."""

    EE_POS = "EE_POS"  # 3 Dim EE Pos
    EE_EULER = "EE_EULER"  # 3 Dim EE Euler
    EE_QUAT = "EE_QUAT"  # 4 Dim EE Quat
    EE_ROT6D = "EE_ROT6D"  # 6 Dim Rot6D
    JOINT_POS = "JOINT_POS"  # 7 x joint
    JOINT_VEL = "JOINT_VEL"  # 7 x joint
    GRIPPER = "GRIPPER"  # 1 x gripper open / close.
    EE_VEL_LIN = "EE_VEL_LIN"
    EE_VEL_ANG = "EE_VEL_ANG"
    MISC = "MISC"  # Other miscellaneous objects


class RobotType(StrEnum):
    """Defines Enum For different robot types"""

    PANDA = "PANDA"
    WIDOWX = "WIDOWX"
    FR3 = "FR3"
    META = "META"
    KUKA = "KUKA"
    JACO = "JACO"
    SAWYER = "SAWYER"
    UR5 = "UR5"
    XARM = "XARM"
    STRETCH = "STRETCH"
    UNKNOWN = "UNKNOWN"


class NormalizationType(StrEnum):
    NONE = "NONE"
    BOUNDS = "BOUNDS"
    GAUSSIAN = "GAUSSIAN"
    BOUNDS_5STDV = "BOUNDS_5STDV"


class DataType(StrEnum):
    BOOL = "BOOL"
    DISCRETE = "DISCRETE"
    CONTINUOUS = "CONTINUOUS"
    CONSTANT = "CONSTANT"


def rmat_to_rot6d(rmat: tf.Tensor) -> tf.Tensor:
    r6 = rmat[..., :2, :]
    r6_0, r6_1 = r6[..., 0, :], r6[..., 1, :]
    r6_flat = tf.concat([r6_0, r6_1], axis=-1)
    return r6_flat


def binarize_gripper_actions(gripper_actions: tf.Tensor) -> tf.Tensor:
    """
    Binarizes the gripper to 0 (open) or 1 (closed).
    Taken from https://github.com/octo-models/octo/blob/main/octo/data/utils/data_utils.py#L292
    """
    open_mask = gripper_actions < 0.05
    closed_mask = gripper_actions > 0.95
    in_between_mask = tf.logical_not(tf.logical_or(open_mask, closed_mask))
    is_closed_float = tf.cast(closed_mask, tf.float32)

    def scan_fn(carry, i):
        return tf.cond(
            in_between_mask[i],
            lambda: tf.cast(carry, tf.float32),  # If we are in between, return the future gripper state.
            lambda: is_closed_float[i],  # If we are not in between, return 1 if closed.
        )

    new_gripper_actions = tf.scan(scan_fn, tf.range(tf.shape(gripper_actions)[0]), is_closed_float[-1], reverse=True)
    return new_gripper_actions


def rel2abs_gripper_actions(
    actions: tf.Tensor,
    threshold: float = 0.1,
):
    """
    Attribution:
    largely borrowed from https://github.com/octo-models/octo/blob/main/octo/data/utils/data_utils.py

    Converts relative gripper actions (+1 for closing, -1 for opening) to absolute gripper actions:
    0 for open, 1 for closed.
    Assumes that the first relative gripper is not redundant (i.e. close when already closed).
    """
    opening_mask = actions < -threshold
    closing_mask = actions > threshold

    # 1 for closing, -1 for opening, 0 for no change
    thresholded_actions = tf.where(opening_mask, -1, tf.where(closing_mask, 1, 0))

    def scan_fn(carry, i):
        # set the gripper action to be the previous one if zero, otherwize set it to be the thresholded action
        return tf.cond(
            thresholded_actions[i] == 0,
            lambda: carry,
            lambda: thresholded_actions[i],
        )

    # Get the action at the first position of change
    first_action = thresholded_actions[tf.argmax(thresholded_actions != 0, axis=0)]
    # If my frist action is 0 (no change) or 1 (closing), then start = open (-1).
    # If my first action is -1 (open), then start = closing (1)
    start = tf.cond(first_action == -1, lambda: 1, lambda: -1)

    # Resulting actions are -1 to 1.
    new_actions = tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), start)
    new_actions = tf.cast(new_actions, tf.float32) / 2 + 0.5
    return new_actions


def gripper_state_from_width(gripper_state: tf.Tensor, max_width: float = 0.079):
    gripper_state = tf.clip_by_value(gripper_state, 0, max_width) / max_width
    return 1 - gripper_state
