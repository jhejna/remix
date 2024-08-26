from typing import Any, Dict

import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tft

from openx.data.utils import RobotType, StateEncoding, gripper_state_from_width, rel2abs_gripper_actions

"""
Note: we follow the 1 for closed, 0 for open gripper convention.

Transform functions are largley taken from:
 https://github.com/rail-berkeley/orca/blob/main/octo/data/oxe/oxe_standardization_transforms.py

OXE datasets not covered:
- tokyo lsmo
- dlr datasets
- asu table top
- uiuc3d
- nyu door opening
- dobbe
"""


def rt1_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    observation = {
        "image": {"agent": ep["observation"]["image"]},
        "state": {
            StateEncoding.EE_POS: ep["observation"]["base_pose_tool_reached"][..., :3],
            StateEncoding.EE_QUAT: ep["observation"]["base_pose_tool_reached"][..., 3:7],
            StateEncoding.EE_EULER: tft.euler.from_quaternion(ep["observation"]["base_pose_tool_reached"][..., 3:7]),
            StateEncoding.GRIPPER: tf.clip_by_value(ep["observation"]["gripper_closed"], 0, 1),
        },
    }

    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = ep["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)[:, None]  # Add back dim

    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"]["world_vector"],
            StateEncoding.EE_EULER: ep["action"]["rotation_delta"],
        },
        "desired_absolute": {StateEncoding.GRIPPER: gripper_action},
    }

    ep["language_instruction"] = ep["observation"]["natural_language_instruction"]
    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.META

    return ep


def kuka_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    # decode compressed state
    eef_value = tf.io.decode_compressed(
        ep["observation"]["clip_function_input/base_pose_tool_reached"],
        compression_type="ZLIB",
    )
    eef_value = tf.reshape(tf.io.decode_raw(eef_value, tf.float32), (-1, 7))
    gripper_value = tf.io.decode_compressed(ep["observation"]["gripper_closed"], compression_type="ZLIB")
    gripper_value = tf.reshape(tf.io.decode_raw(gripper_value, tf.float32), (-1, 1))

    observation = {
        "image": {"agent": ep["observation"]["image"]},
        "state": {
            StateEncoding.EE_POS: eef_value[..., :3],
            StateEncoding.EE_QUAT: eef_value[..., 3:7],
            StateEncoding.EE_EULER: tft.euler.from_quaternion(eef_value[..., 3:7]),
            StateEncoding.GRIPPER: tf.clip_by_value(gripper_value, 0, 1),
        },
    }

    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = ep["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)[:, None]
    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"]["world_vector"],
            StateEncoding.EE_EULER: ep["action"]["rotation_delta"],
        },
        "desired_absolute": {StateEncoding.GRIPPER: gripper_action},
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.KUKA

    return ep


def taco_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    observation = {
        "image": {"agent": ep["observation"]["rgb_static"], "wrist": ep["observation"]["rgb_gripper"]},
        "state": {
            StateEncoding.EE_POS: ep["observation"]["robot_obs"][..., :3],
            StateEncoding.EE_EULER: ep["observation"]["robot_obs"][..., 3:6],
            StateEncoding.EE_QUAT: tft.quaternion.from_euler(ep["observation"]["robot_obs"][..., 3:6]),
            StateEncoding.GRIPPER: gripper_state_from_width(ep["observation"]["robot_obs"][..., 6:7], max_width=0.0806),
        },
    }

    # make gripper action absolute action, +1 = open, 0 = close
    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"]["rel_actions_world"][..., :3],
            StateEncoding.EE_EULER: ep["action"]["rel_actions_world"][..., 3:6],
        },
        "desired_absolute": {
            StateEncoding.GRIPPER: 1 - tf.clip_by_value((ep["action"]["rel_actions_world"][..., 6:7] + 1) / 2, 0, 1)
        },
    }

    ep["language_instruction"] = ep["observation"]["natural_language_instruction"]
    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.PANDA

    return ep


def jaco_play_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, 0 = open, 1 = nothing, 2 = close
    gripper_action = ep["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)[:, None]

    observation = {
        "image": {"agent": ep["observation"]["image"], "wrist": ep["observation"]["image_wrist"]},
        "state": {
            StateEncoding.EE_POS: ep["observation"]["end_effector_cartesian_pos"][..., :3],
            StateEncoding.EE_QUAT: ep["observation"]["end_effector_cartesian_pos"][..., 3:7],
            StateEncoding.EE_EULER: tft.euler.from_quaternion(
                ep["observation"]["end_effector_cartesian_pos"][..., 3:7]
            ),
        },
    }

    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"]["world_vector"],
            StateEncoding.EE_EULER: tf.zeros_like(ep["action"]["world_vector"]),
        },
        "desired_absolute": {StateEncoding.GRIPPER: gripper_action},
    }

    ep["language_instruction"] = ep["observation"]["natural_language_instruction"]
    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.JACO

    return ep


def berkeley_cable_routing_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    observation = {
        "image": {"agent": ep["observation"]["image"], "wrist": ep["observation"]["wrist45_image"]},
        "state": {
            StateEncoding.JOINT_POS: ep["observation"]["robot_state"],
            StateEncoding.GRIPPER: tf.ones_like(ep["action"]["world_vector"][:, :1]),
        },
    }

    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"]["world_vector"],
            StateEncoding.EE_EULER: ep["action"]["rotation_delta"],
        },
        "desired_absolute": {
            StateEncoding.GRIPPER: tf.ones_like(ep["action"]["world_vector"][:, :1]),
        },
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.PANDA

    return ep


def roboturk_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    observation = {
        "image": {"agent": ep["observation"]["front_rgb"]},
        "state": {},  # No state observation provided :(
    }

    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"]["world_vector"],
            StateEncoding.EE_EULER: ep["action"]["rotation_delta"],
        },
        "desired_absolute": {StateEncoding.GRIPPER: tf.clip_by_value(ep["action"]["gripper_closedness_action"], 0, 1)},
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.SAWYER

    return ep


def viola_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    gripper_action = ep["action"]["gripper_closedness_action"][:, None]
    gripper_action = tf.clip_by_value(gripper_action, 0, 1)

    observation = {
        "image": {"agent": ep["observation"]["agentview_rgb"], "wrist": ep["observation"]["eye_in_hand_rgb"]},
        "state": {
            # NOTE: EE coordiantes can be taken from 4x4 matrix in ee_states
            StateEncoding.JOINT_POS: ep["observation"]["joint_states"],
            StateEncoding.GRIPPER: gripper_state_from_width(ep["observation"]["gripper_states"], max_width=0.075),
        },
    }

    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"]["world_vector"],
            StateEncoding.EE_EULER: ep["action"]["rotation_delta"],
        },
        "desired_absolute": {
            StateEncoding.GRIPPER: gripper_action,
        },
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.PANDA

    return ep


def berkeley_autolab_ur5_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    # TODO: should check the gripper.
    gripper_action = ep["action"]["gripper_closedness_action"]
    gripper_action = rel2abs_gripper_actions(gripper_action)[:, None]

    observation = {
        "image": {"agent": ep["observation"]["image"], "wrist": ep["observation"]["hand_image"]},
        "state": {
            StateEncoding.EE_POS: ep["observation"]["robot_state"][..., 6:9],
            StateEncoding.EE_QUAT: ep["observation"]["robot_state"][..., 9:13],
            StateEncoding.GRIPPER: ep["observation"]["robot_state"][..., 13:14],
            StateEncoding.EE_EULER: tft.euler.from_quaternion(ep["observation"]["robot_state"][..., 9:13]),
        },
    }

    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"]["world_vector"],
            StateEncoding.EE_EULER: ep["action"]["rotation_delta"],
        },
        "desired_absolute": {StateEncoding.GRIPPER: gripper_action},
    }

    ep["language_instruction"] = ep["observation"]["natural_language_instruction"]
    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.UR5
    return ep


def toto_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    observation = {
        "image": {"agent": ep["observation"]["image"]},
        "state": {
            StateEncoding.JOINT_POS: ep["observation"]["state"],
        },
    }

    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"]["world_vector"],
            StateEncoding.EE_EULER: ep["action"]["rotation_delta"],
        },
        "desired_absolute": {
            StateEncoding.GRIPPER: 1
            - tf.clip_by_value(tf.cast(ep["action"]["open_gripper"][:, None], tf.float32), 0, 1),
        },
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.PANDA
    return ep


def language_table_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    observation = {
        "image": {"agent": ep["observation"]["rgb"]},
        "state": {
            StateEncoding.EE_POS: tf.concat(
                (
                    ep["observation"]["effector_translation"],
                    tf.zeros_like(ep["observation"]["effector_translation"][..., :1]),
                ),
                axis=-1,
            ),
            # Pretend gripper is always closed since we are holding something.
            StateEncoding.GRIPPER: tf.ones_like(ep["observation"]["effector_translation"][..., :1]),
        },
    }

    desired_ee_pos_delta = tf.concat((ep["action"], tf.zeros_like(ep["action"][..., :1])), axis=-1)
    action = {
        "desired_delta": {
            StateEncoding.EE_POS: desired_ee_pos_delta,
            StateEncoding.EE_EULER: tf.zeros_like(desired_ee_pos_delta),
        },
        "desired_absolute": {
            StateEncoding.GRIPPER: tf.zeros_like(desired_ee_pos_delta[..., :1]),  # default to "open" gripper
        },
    }

    # decode language instruction
    instruction_bytes = ep["observation"]["instruction"]
    instruction_encoded = tf.strings.unicode_encode(instruction_bytes, output_encoding="UTF-8")
    # Remove trailing padding --> convert RaggedTensor to regular Tensor.
    ep["language_instruction"] = tf.strings.split(instruction_encoded, "\x00")[:, :1].to_tensor()[:, 0]

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.XARM
    return ep


def stanford_hydra_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    observation = {
        "image": {"agent": ep["observation"]["image"], "wrist": ep["observation"]["wrist_image"]},
        "state": {
            StateEncoding.EE_POS: ep["observation"]["state"][..., :3],
            StateEncoding.EE_QUAT: ep["observation"]["state"][..., 3:7],
            StateEncoding.EE_EULER: ep["observation"]["state"][..., 7:10],
            StateEncoding.GRIPPER: gripper_state_from_width(ep["observation"]["state"][..., -3:-2], max_width=0.082),
        },
    }

    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"][..., :3],
            StateEncoding.EE_EULER: ep["action"][..., 3:6],
        },
        "desired_absolute": {
            StateEncoding.GRIPPER: ep["action"][..., -1:],
        },
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.PANDA
    return ep


def austin_buds_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    observation = {
        "image": {"agent": ep["observation"]["image"], "wrist": ep["observation"]["wrist_image"]},
        "state": {
            # NOTE: EE coordiantes can be taken from 4x4 matrix in state[8:]
            StateEncoding.JOINT_POS: ep["observation"]["state"][..., :7],
            StateEncoding.GRIPPER: gripper_state_from_width(ep["observation"]["state"][..., 7:8], max_width=0.079),
        },
    }

    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"][..., :3],
            StateEncoding.EE_EULER: ep["action"][..., 3:6],
        },
        "desired_absolute": {StateEncoding.GRIPPER: tf.clip_by_value(ep["action"][..., -1:], 0, 1)},
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.PANDA
    return ep


def nyu_franka_play_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    observation = {
        "image": {"agent": ep["observation"]["image"]},
        "state": {
            StateEncoding.JOINT_POS: ep["observation"]["state"][..., :7],
            StateEncoding.EE_POS: ep["observation"]["state"][..., 7:10],
            StateEncoding.EE_EULER: ep["observation"]["state"][..., 10:13],
            StateEncoding.EE_QUAT: tft.quaternion.from_euler(ep["observation"]["state"][..., 10:13]),
        },
    }

    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"][..., 7:10],
            StateEncoding.EE_EULER: ep["action"][..., 10:13],
        },
        "desired_absolute": {
            StateEncoding.GRIPPER: 1 - tf.clip_by_value(ep["action"][..., -2:-1], 0, 1)  # Invert gripper
        },
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.PANDA
    return ep


def maniskill_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    observation = {
        "image": {"agent": ep["observation"]["image"], "wrist": ep["observation"]["wrist_image"]},
        "state": {
            StateEncoding.EE_POS: ep["observation"]["tcp_pose"][..., :3],
            StateEncoding.EE_QUAT: ep["observation"]["tcp_pose"][..., 3:7],
            StateEncoding.EE_EULER: tft.euler.from_quaternion(ep["observation"]["tcp_pose"][..., 3:7]),
            StateEncoding.GRIPPER: gripper_state_from_width(ep["observation"]["state"][..., 7:8], max_width=0.04),
        },
    }

    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"][..., :3],
            StateEncoding.EE_EULER: ep["action"][..., 3:6],
        },
        "desired_absolute": {
            StateEncoding.GRIPPER: 1 - tf.clip_by_value(ep["action"][..., -1:], 0, 1)  # Invert gripper
        },
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.PANDA
    return ep


def furniture_bench_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    observation = {
        "image": {"agent": ep["observation"]["image"], "wrist": ep["observation"]["wrist_image"]},
        "state": {
            StateEncoding.EE_POS: ep["observation"]["state"][..., :3],
            StateEncoding.EE_QUAT: ep["observation"]["state"][..., 3:7],
            StateEncoding.EE_EULER: tft.euler.from_quaternion(ep["observation"]["state"][..., 3:7]),
            # NOTE: gripper might not be correctly formatted.
            StateEncoding.GRIPPER: gripper_state_from_width(ep["observation"]["state"][..., -1:], max_width=0.065),
        },
    }

    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"][..., :3],
            StateEncoding.EE_EULER: tft.euler.from_quaternion(ep["action"][..., 3:7]),
        },
        "desired_absolute": {StateEncoding.GRIPPER: tf.clip_by_value(ep["action"][..., -1:], 0, 1)},
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.PANDA
    return ep


def cmu_franka_exploration_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    observation = {
        "image": {"agent": ep["observation"]["highres_image"]},
        "state": {},  # No state given :(
    }

    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"][..., :3],
            StateEncoding.EE_EULER: ep["action"][..., 3:6],
        },
        "desired_absolute": {StateEncoding.GRIPPER: 1 - tf.clip_by_value(ep["action"][..., 6:7], 0, 1)},  # Invert
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.PANDA
    return ep


def ucsd_kitchen_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    observation = {
        "image": {"agent": ep["observation"]["image"]},
        "state": {StateEncoding.JOINT_POS: ep["observation"]["state"][..., :7]},
    }

    # Actions in this dataset are huge for some reason.... divide by 1000.
    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"][..., :3] / 1000,
            StateEncoding.EE_EULER: ep["action"][..., 3:6] / 1000,
        },
        "desired_absolute": {StateEncoding.GRIPPER: 1 - tf.clip_by_value(ep["action"][..., 6:7], 0, 1)},  # Invert
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.XARM
    return ep


def austin_sailor_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    observation = {
        "image": {"agent": ep["observation"]["image"], "wrist": ep["observation"]["wrist_image"]},
        "state": {
            StateEncoding.EE_POS: ep["observation"]["state"][..., :3],
            StateEncoding.EE_QUAT: ep["observation"]["state"][..., 3:7],
            StateEncoding.EE_EULER: tft.euler.from_quaternion(ep["observation"]["state"][..., 3:7]),
            StateEncoding.JOINT_POS: ep["observation"]["state_joint"],
            StateEncoding.GRIPPER: ep["observation"]["state"][..., 7:8],
        },
    }

    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"][..., :3],
            StateEncoding.EE_EULER: ep["action"][..., 3:6],
        },
        "desired_absolute": {StateEncoding.GRIPPER: gripper_state_from_width(ep["action"][..., -1:], max_width=0.079)},
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.PANDA
    return ep


def austin_sirius_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    observation = {
        "image": {"agent": ep["observation"]["image"], "wrist": ep["observation"]["wrist_image"]},
        "state": {
            # NOTE: Can get EE state from matrix
            StateEncoding.JOINT_POS: ep["observation"]["state_joint"],
            StateEncoding.GRIPPER: gripper_state_from_width(ep["observation"]["state_gripper"], max_width=0.079),
        },
    }

    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"][..., :3],
            StateEncoding.EE_EULER: ep["action"][..., 3:6],
        },
        "desired_absolute": {StateEncoding.GRIPPER: tf.clip_by_value(ep["action"][..., -1:], 0, 1)},
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.PANDA
    return ep


def bc_z_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    observation = {
        "image": {"agent": ep["observation"]["image"]},
        "state": {
            StateEncoding.EE_POS: ep["observation"]["present/xyz"],
            StateEncoding.EE_EULER: ep["observation"]["present/axis_angle"],
            StateEncoding.EE_QUAT: tft.quaternion.from_euler(ep["observation"]["present/axis_angle"]),
            StateEncoding.GRIPPER: ep["observation"]["present/sensed_close"],
        },
    }

    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"]["future/xyz_residual"][:, :3],
            StateEncoding.EE_EULER: ep["action"]["future/axis_angle_residual"][:, :3],
        },
        "desired_absolute": {StateEncoding.GRIPPER: tf.cast(ep["action"]["future/target_close"][:, :1], tf.float32)},
    }

    ep["language_instruction"] = ep["observation"]["natural_language_instruction"]
    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.META
    return ep


def berkeley_mvp_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    observation = {
        "image": {"wrist": ep["observation"]["hand_image"]},
        "state": {
            StateEncoding.EE_POS: ep["observation"]["pose"][..., :3],
            StateEncoding.EE_EULER: tft.euler.from_quaternion(ep["observation"]["pose"][..., 3:7]),
            StateEncoding.EE_QUAT: ep["observation"]["pose"][..., 3:7],
            StateEncoding.JOINT_POS: ep["observation"]["joint_pos"],
            StateEncoding.GRIPPER: ep["observation"]["gripper"][..., None],
        },
    }

    action = {
        "desired_delta": {
            StateEncoding.JOINT_POS: ep["action"][..., :7],
        },
        "desired_absolute": {
            StateEncoding.GRIPPER: ep["action"][..., 7:],
        },
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.PANDA
    return ep


def berkeley_rpt_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    # relabel actions to convert from 30Hz to 10Hz
    factor = 3
    ep = tf.nest.map_structure(lambda x: x[::factor], ep)

    observation = {
        "image": {"wrist": ep["observation"]["image"]},
        "state": {
            StateEncoding.JOINT_POS: ep["observation"]["joint_pos"],
            StateEncoding.GRIPPER: ep["observation"]["gripper"][..., None],
        },
    }

    # recompute actions for downsampled sequence
    joint_actions = ep["observation"]["joint_pos"][1:, :7] - ep["observation"]["joint_pos"][:-1, :7]

    action = {
        "desired_delta": {
            StateEncoding.JOINT_POS: tf.concat((joint_actions, tf.zeros_like(joint_actions[-1:])), axis=0)
        },
        "desired_absolute": {
            StateEncoding.GRIPPER: ep["action"][..., 7:],
        },
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.PANDA
    return ep


def kaist_nonprehensible_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    observation = {
        "image": {"agent": ep["observation"]["image"]},
        "state": {
            StateEncoding.EE_POS: ep["observation"]["state"][..., 14:17],
            StateEncoding.EE_QUAT: ep["observation"]["state"][..., 17:21],
            StateEncoding.EE_EULER: tft.euler.from_quaternion(ep["observation"]["state"][..., 17:21]),
            StateEncoding.GRIPPER: tf.ones_like(ep["observation"]["state"][..., -1:]),
        },
    }

    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"][..., :3],
            StateEncoding.EE_EULER: ep["action"][..., 3:6],
        },
        "desired_absolute": {
            StateEncoding.GRIPPER: tf.ones_like(ep["action"][..., -1:]),
        },
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.PANDA
    return ep


def dlr_edan_shared_control_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    observation = {
        "image": {"agent": ep["observation"]["image"]},
        "state": {
            StateEncoding.EE_POS: ep["observation"]["state"][..., :3],
            StateEncoding.EE_EULER: ep["observation"]["state"][
                ..., 3:6
            ],  # calculated with scipy Rotation.as_euler(=\"zxy\")
            StateEncoding.EE_QUAT: tft.quaternion.from_euler(ep["observation"]["state"][..., 3:6]),
            StateEncoding.GRIPPER: ep["observation"]["state"][..., -1:],
        },
    }

    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"][..., :3],
            StateEncoding.EE_EULER: ep["action"][..., 3:6],
        },
        "desired_absolute": {
            StateEncoding.GRIPPER: ep["action"][..., 6:7],
        },
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.UNKNOWN
    return ep


def robocook_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    observation = {
        "image": {
            "agent": ep["observation"]["image_1"],
        },  # ep["observation"]["image_2"]
        "state": {
            StateEncoding.EE_POS: ep["observation"]["state"][..., :3],
            StateEncoding.EE_EULER: ep["observation"]["state"][..., 3:6],
            StateEncoding.EE_QUAT: tft.quaternion.from_euler(ep["observation"]["state"][..., 3:6]),
            StateEncoding.GRIPPER: gripper_state_from_width(ep["observation"]["state"][..., -1:], max_width=0.05),
        },
    }

    gripper_action = rel2abs_gripper_actions(-ep["action"][:, 6], threshold=0.001)[:, None]
    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"][..., :3],
            StateEncoding.EE_EULER: ep["action"][..., 3:6],
        },
        "desired_absolute": {
            StateEncoding.GRIPPER: gripper_action,
        },
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.PANDA
    return ep


def iamlab_pick_insert_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    observation = {
        "image": {"agent": ep["observation"]["image"], "wrist": ep["observation"]["wrist_image"]},
        "state": {
            StateEncoding.JOINT_POS: ep["observation"]["state"][..., :7],
            StateEncoding.GRIPPER: tf.clip_by_value(1 - ep["observation"]["state"][..., 7:8], 0, 1),
        },
    }

    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"][..., :3],
            StateEncoding.EE_EULER: tft.euler.from_quaternion(ep["action"][..., 3:7]),
        },
        "desired_absolute": {StateEncoding.GRIPPER: tf.clip_by_value(1 - ep["action"][..., 7:8], 0, 1)},
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.PANDA
    return ep


def utaustin_mutex_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    observation = {
        "image": {"agent": ep["observation"]["image"], "wrist": ep["observation"]["wrist_image"]},
        "state": {
            # NOTE: EE coordiantes can be taken from 4x4 matrix in ee_states
            StateEncoding.JOINT_POS: ep["observation"]["state"][..., :7],
            StateEncoding.GRIPPER: gripper_state_from_width(ep["observation"]["state"][..., 7:8], max_width=0.079),
        },
    }

    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"][..., :3],
            StateEncoding.EE_EULER: ep["action"][..., 3:6],
        },
        "desired_absolute": {StateEncoding.GRIPPER: tf.clip_by_value(ep["action"][..., 6:7], 0, 1)},
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.PANDA
    return ep


def berkeley_fanuc_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    observation = {
        "image": {"agent": ep["observation"]["image"], "wrist": ep["observation"]["wrist_image"]},
        "state": {
            # NOTE: EE coordiantes can be taken from 4x4 matrix in ee_states
            StateEncoding.EE_POS: ep["observation"]["end_effector_state"][..., :3],
            StateEncoding.EE_QUAT: ep["observation"]["end_effector_state"][..., 3:7],
            StateEncoding.EE_EULER: tft.euler.from_quaternion(ep["observation"]["end_effector_state"][..., 3:7]),
            StateEncoding.GRIPPER: ep["observation"]["state"][..., 6:7],
        },
    }

    # Shift the gripper action to be the state at the next step
    gripper_action = tf.roll(ep["observation"]["state"][..., 6:7], shift=-1, axis=0)
    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"][..., :3],
            StateEncoding.EE_EULER: ep["action"][..., 3:6],
        },
        "desired_absolute": {StateEncoding.GRIPPER: gripper_action},
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.UNKNOWN
    return ep


def playfusion_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    observation = {
        "image": {"agent": ep["observation"]["image"]},
        "state": {
            StateEncoding.JOINT_POS: ep["observation"]["state"][..., :7],
            StateEncoding.GRIPPER: gripper_state_from_width(ep["observation"]["state"][..., 7:8], max_width=0.085),
        },
    }

    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"][..., :3],
            StateEncoding.EE_EULER: tft.euler.from_quaternion(ep["action"][..., 3:7]),
        },
        "desired_absolute": {
            StateEncoding.GRIPPER: 1 - ep["action"][..., 7:8],
        },
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.PANDA
    return ep


def cmu_stretch_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    observation = {
        "image": {"agent": ep["observation"]["image"]},
        "state": {
            StateEncoding.EE_POS: ep["observation"]["state"][..., :3],
            StateEncoding.GRIPPER: ep["observation"]["state"][..., 3:4],
        },
    }

    action = {
        "desired_delta": {StateEncoding.EE_POS: ep["action"][..., :3], StateEncoding.EE_EULER: ep["action"][..., 3:6]},
        "desired_absolute": {
            StateEncoding.GRIPPER: 1 - ep["action"][..., 6:7],
        },
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.STRETCH
    return ep


def rh20t_dataset_transform(ep: Dict[str, Any]) -> Dict[str, Any]:
    observation = {
        "image": {"agent": ep["observation"]["image_front"], "wrist": ep["observation"]["image_wrist"]},
        "state": {
            StateEncoding.EE_POS: ep["observation"]["tcp_base"][..., :3],
            StateEncoding.EE_EULER: ep["observation"]["tcp_base"][..., 3:6],
            StateEncoding.EE_QUAT: tft.quaternion.from_euler(ep["observation"]["tcp_base"][..., 3:6]),
            StateEncoding.GRIPPER: gripper_state_from_width(
                ep["observation"]["gripper_width"][..., None], max_width=80.0
            ),
        },
    }

    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"]["tcp_base"][..., :3],
            StateEncoding.EE_EULER: ep["action"]["tcp_base"][..., 3:6],
        },
        "desired_absolute": {StateEncoding.GRIPPER: 1 - tf.cast(ep["action"]["gripper"], tf.float32)[..., None]},
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.UNKNOWN
    return ep
