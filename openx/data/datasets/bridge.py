from typing import Dict

import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tft

from openx.data.utils import RobotType, StateEncoding, binarize_gripper_actions

# Define groups on the BRIDGE dataset metadata
# Split into 32 domains
FULL_BRIDGE_DOMAINS = {
    "toykitchen2": 0,
    "datacol2_tabletop_dark_wood": 1,
    "toykitchen1": 2,
    "toykitchen6": 3,
    "datacol2_toykitchen7": 4,
    "datacol2_toykitchen2": 5,
    "toykitchen7": 6,
    "datacol2_folding_table": 7,
    "datacol1_toykitchen6": 8,
    "datacol2_robot_desk": 9,
    "datacol2_toykitchen6": 10,
    "deepthought_folding_table": 11,
    "datacol2_laundry_machine": 12,
    "datacol2_toykitchen5": 13,
    "deepthought_toykitchen2": 14,
    "deepthought_robot_desk": 15,  ## Everything below here less than 1000
    "tabletop_dark_wood": 16,
    "datacol2_toysink2": 17,
    "toykitchen2_room8052": 18,
    "deepthought_toykitchen1": 19,
    "toykitchen5": 13,
    # Missing 20, 24 28,
    "datacol2_folding_table_white_tray": 20,
    "toysink3_bww": 21,  ## Under here is under 500
    "datacol2_toykitchen1": 22,
    "toysink2_bww": 17,  # REASSIGNED
    "toysink1_room8052": 23,
    "tool_chest": 24,
    "toysink5": 25,
    "datacol1_toykitchen1": 19,  # REASSIGNED
    "whiteboard": 26,
    "toykitchen4": 27,  # REASSIGNED
    "toysink4": 28,  # REASSIGNED
    "laundry_machine": 12,  # REASSIGNED
    "toysink3": 21,  # REASSIGNED
    "toysink1": 23,  # REASSIGNED
    "toykitchen3": 29,  # REASSIGNED
    "realkitchen1_dishwasher": 30,  # REASSIGNED
    "minsky_folding_table_white_tray": 20,  # REASSIGNED
    "tabletop_light_wood": 31,  # REASSIGNED
    "tabletop_white": 31,  # REASSIGNED
    "realkitchen1_counter": 31,  # REASSIGNED
    "datacol2_toykitchen7_white_tray": 20,  # REASSIGNED
}

FULL_BRIDGE_WEIGHTS = {
    1: 0.09452740566011847,
    2: 0.06930769751474122,
    12: 0.025829542703360006,
    7: 0.03852202935865529,
    24: 0.004715240802322738,
    22: 0.011554535370099445,
    11: 0.027280948951358987,
    6: 0.03280300087389292,
    0: 0.18728751077120287,
    3: 0.06940527104401847,
    27: 0.0037193809691368827,
    16: 0.02199856201011228,
    5: 0.04329276510575446,
    19: 0.018682281683802,
    8: 0.0360662255936587,
    9: 0.025810027997504557,
    17: 0.02257485566740599,
    21: 0.01235829731752073,
    18: 0.010835540426237767,
    4: 0.07133783675826585,
    14: 0.025331307869488087,
    10: 0.023943934250077296,
    20: 0.03785669985589609,
    13: 0.033736657582164535,
    31: 0.004647549166386651,
    15: 0.019783642895518875,
    23: 0.009794552835761186,
    30: 0.0020264802361767277,
    25: 0.0040541801414694205,
    26: 0.006774042270072552,
    29: 0.0012440624982848402,
    28: 0.0028979338195340987,
}

# Split into 16 domains
COMPRESSED_BRIDGE_DOMAINS = {
    "toykitchen2": 0,
    "datacol2_tabletop_dark_wood": 1,
    "toykitchen1": 2,
    "toykitchen6": 3,
    "datacol2_toykitchen7": 4,
    "datacol2_toykitchen2": 5,
    "toykitchen7": 4,
    "datacol2_folding_table": 6,
    "datacol1_toykitchen6": 7,
    "datacol2_robot_desk": 8,
    "datacol2_toykitchen6": 7,
    "deepthought_folding_table": 6,
    "datacol2_laundry_machine": 9,
    "datacol2_toykitchen5": 10,
    "deepthought_toykitchen2": 5,
    "deepthought_robot_desk": 8,  ## Everything below here less than 1000
    "tabletop_dark_wood": 1,
    "datacol2_toysink2": 11,
    "toykitchen2_room8052": 5,
    "deepthought_toykitchen1": 2,
    "toykitchen5": 10,
    "datacol2_folding_table_white_tray": 12,
    "toysink3_bww": 13,  ## Under here is under 500
    "datacol2_toykitchen1": 2,
    "toysink2_bww": 11,  # REASSIGNED
    "toysink1_room8052": 14,
    "tool_chest": 9,
    "toysink5": 15,
    "datacol1_toykitchen1": 2,  # REASSIGNED
    "whiteboard": 12,
    "toykitchen4": 15,  # REASSIGNED
    "toysink4": 15,  # REASSIGNED
    "laundry_machine": 9,  # REASSIGNED
    "toysink3": 13,  # REASSIGNED
    "toysink1": 14,  # REASSIGNED
    "toykitchen3": 15,  # REASSIGNED
    "realkitchen1_dishwasher": 9,  # REASSIGNED
    "minsky_folding_table_white_tray": 12,  # REASSIGNED
    "tabletop_light_wood": 12,  # REASSIGNED
    "tabletop_white": 12,  # REASSIGNED
    "realkitchen1_counter": 12,  # REASSIGNED
    "datacol2_toykitchen7_white_tray": 12,  # REASSIGNED
}

COMPRESSED_BRIDGE_WEIGHTS = {
    1: 0.11652596767023074,
    2: 0.09954451456864268,
    9: 0.03257126374185947,
    6: 0.06580297831001428,
    4: 0.10414083763215877,
    0: 0.18728751077120287,
    3: 0.06940527104401847,
    15: 0.011915557428425242,
    5: 0.0794596134014803,
    7: 0.06001015984373599,
    8: 0.04559367089302343,
    11: 0.02257485566740599,
    13: 0.01235829731752073,
    12: 0.049278291292355295,
    10: 0.033736657582164535,
    14: 0.009794552835761186,
}

assert len(set(list(FULL_BRIDGE_DOMAINS.values()))) == 32
assert len(set(list(COMPRESSED_BRIDGE_DOMAINS.values()))) == 16

keys, values = zip(*sorted([(k, v) for k, v in FULL_BRIDGE_DOMAINS.items()]), strict=False)
FULL_DOMAIN_TABLE = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(keys),
        values=tf.constant(values, dtype=tf.int32),
    ),
    default_value=FULL_BRIDGE_DOMAINS["realkitchen1_counter"],
)

keys, values = zip(*sorted([(k, v) for k, v in COMPRESSED_BRIDGE_DOMAINS.items()]), strict=False)
COMPRESSED_DOMAIN_TABLE = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(keys),
        values=tf.constant(values, dtype=tf.int32),
    ),
    default_value=COMPRESSED_BRIDGE_DOMAINS["realkitchen1_counter"],
)


def bridge_dataset_transform(ep: Dict):
    # This is currently for the incorrect bridge dataset.
    observation = {
        "image": {"agent": ep["observation"]["image_0"]},
        "state": {
            StateEncoding.EE_POS: ep["observation"]["state"][..., :3],
            StateEncoding.EE_EULER: ep["observation"]["state"][..., 3:6],
            StateEncoding.EE_QUAT: tft.quaternion.from_euler(ep["observation"]["state"][..., 3:6]),
            StateEncoding.GRIPPER: 1 - tf.clip_by_value(ep["observation"]["state"][..., -1:], 0, 1),
        },
    }

    state_minus_gripper = ep["observation"]["state"][:, :6]
    achieved_delta = tf.roll(state_minus_gripper, shift=-1, axis=0) - state_minus_gripper
    gripper = binarize_gripper_actions(1 - tf.clip_by_value(ep["action"][:, -1:], 0, 1))  # Bridge is backwards!

    # NOTE: temporary shim to make this work with training on desired delta
    # later desired delta should be set to the commented version
    action = {
        # "desired_delta": {
        #     StateEncoding.EE_POS: ep["action"][:, :3],
        #     StateEncoding.EE_EULER: ep["action"][:, 3:6],
        # },
        "desired_delta": {
            StateEncoding.EE_POS: achieved_delta[:, :3],
            StateEncoding.EE_EULER: achieved_delta[:, 3:6],
        },
        "desired_absolute": {StateEncoding.GRIPPER: gripper},
        "achieved_delta": {
            StateEncoding.EE_POS: achieved_delta[:, :3],
            StateEncoding.EE_EULER: achieved_delta[:, 3:6],
        },
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.WIDOWX

    # Assign a scene_id
    parts = tf.strings.split(ep["episode_metadata"]["file_path"][0], "/")
    dataset_name = parts[4]
    string_key = tf.cond(dataset_name == "bridge_data_v2" or dataset_name == "rss", lambda: parts[5], lambda: parts[6])
    scene_id = FULL_DOMAIN_TABLE.lookup(string_key)
    scene_id_compressed = COMPRESSED_DOMAIN_TABLE.lookup(string_key)
    ep_len = ep["ep_len"][0]
    ep["full_scene_id"] = tf.repeat(scene_id, ep_len)
    ep["compressed_scene_id"] = tf.repeat(scene_id_compressed, ep_len)
    ep["full_num_scenes"] = tf.repeat(32, ep_len)
    ep["compressed_num_scenes"] = tf.repeat(16, ep_len)

    return ep
