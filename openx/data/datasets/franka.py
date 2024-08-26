from typing import Dict

import tensorflow as tf

from openx.data.utils import RobotType, StateEncoding


def franka_dataset_transform(ep: Dict):
    observation = {
        "image": {"agent": ep["observation"]["agent_image"]},
        "state": {
            StateEncoding.EE_POS: ep["observation"]["state"]["ee_pos"],
            StateEncoding.EE_QUAT: ep["observation"]["state"]["ee_quat"],
            StateEncoding.GRIPPER: tf.clip_by_value(ep["observation"]["state"]["gripper_pos"], 0, 1),
        },
    }

    achieved_delta_pos = (
        tf.roll(ep["observation"]["state"]["ee_pos"], shift=-1, axis=0) - ep["observation"]["state"]["ee_pos"]
    )
    achieved_delta_quat = (
        tf.roll(ep["observation"]["state"]["ee_quat"], shift=-1, axis=0) - ep["observation"]["state"]["ee_quat"]
    )
    gripper = tf.clip_by_value((ep["action"][:, -1:] + 1) / 2, 0, 1)
    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"][:, :3],
            StateEncoding.EE_EULER: ep["action"][:, 3:6],
        },
        "desired_absolute": {StateEncoding.GRIPPER: gripper},
        "achieved_delta": {
            StateEncoding.EE_POS: achieved_delta_pos,
            StateEncoding.EE_QUAT: achieved_delta_quat,  # using quat instead of euler here
        },
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.PANDA

    return ep
