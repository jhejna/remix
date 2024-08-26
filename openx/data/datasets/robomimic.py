from typing import Dict

from openx.data.utils import RobotType, StateEncoding


def robomimic_dataset_transform(ep: Dict):
    # Optional: rescale data to be in absolute space. Removing for now.
    # action_max = np.array((0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 1), dtype=np.float32) # gripper is 0 to 1
    # action_min = np.array((-0.05, -0.05, -0.05, -0.5, -0.5, -0.5, 0), dtype=np.float32)
    # action = (ep["action"] + 1) / 2 * (action_max - action_min) + action_min
    action = ep["action"]
    delta_ee_pos, delta_ee_euler, gripper_action = action[..., :3], action[..., 3:6], action[..., -1:]
    # ee_rmat = tfg.rotation_matrix_3d.from_quaternion(ep["observation"]["state"]["ee_quat"])
    # delta_ee_rmat = tfg.rotation_matrix_3d.from_euler(delta_ee_euler)

    observation = {
        "image": {"agent": ep["observation"]["agent_image"], "wrist": ep["observation"]["wrist_image"]},
        "state": {
            StateEncoding.EE_POS: ep["observation"]["state"]["ee_pos"],
            StateEncoding.EE_QUAT: ep["observation"]["state"]["ee_quat"],
            # StateEncoding.EE_ROT6D: rmat_to_rot6d(ee_rmat),
            StateEncoding.GRIPPER: ep["observation"]["state"]["gripper_qpos"][..., :1],
            StateEncoding.JOINT_POS: ep["observation"]["state"]["joint_pos"],
            StateEncoding.JOINT_VEL: ep["observation"]["state"]["joint_vel"],
            StateEncoding.MISC: ep["observation"]["state"]["object"],
        },
    }

    action = {
        "desired_delta": {
            StateEncoding.EE_POS: delta_ee_pos,
            StateEncoding.EE_EULER: delta_ee_euler,
            # StateEncoding.EE_ROT6D: rmat_to_rot6d(delta_ee_rmat),
        },
        "desired_absolute": {
            StateEncoding.GRIPPER: gripper_action,
            # StateEncoding.EE_POS: ep["observation"]["state"]["ee_pos"] + delta_ee_pos,
            # StateEncoding.EE_ROT6D: rmat_to_rot6d(ee_rmat * delta_ee_rmat),
        },
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.PANDA
    return ep
