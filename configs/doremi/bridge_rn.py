import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import optax
from dataset_paths import source_paths
from ml_collections import ConfigDict

from openx.data.datasets.bridge import FULL_BRIDGE_WEIGHTS, bridge_dataset_transform
from openx.data.utils import DataType, NormalizationType, StateEncoding
from openx.networks.action_heads import DiscreteActionHead
from openx.networks.core import Model
from openx.networks.mlp import MLP, Concatenate
from openx.networks.resnet import ResNet18, ResNet34, ResNet50
from openx.utils.spec import ModuleSpec


def get_config(config_str: str = "img,34,us_bridge_source"):
    # By default returns the bridge config.
    data_type, resnet_class, bridge_name = config_str.split(",")
    source_path = source_paths[bridge_name]
    assert data_type in {"state", "img"}
    assert resnet_class in {"18", "34", "50"}

    resnet_class = {"18": ResNet18, "34": ResNet34, "50": ResNet50}[resnet_class]

    structure = {
        "observation": {
            "image": {
                "agent": (224, 288),  # Height x width
            },
            "state": {
                StateEncoding.EE_POS: NormalizationType.NONE,
                StateEncoding.EE_EULER: NormalizationType.NONE,
                StateEncoding.GRIPPER: NormalizationType.NONE,
            },
        },
        "action": {
            "achieved_delta": {
                StateEncoding.EE_POS: NormalizationType.BOUNDS,
                StateEncoding.EE_EULER: NormalizationType.BOUNDS,
            },
            "desired_absolute": {StateEncoding.GRIPPER: NormalizationType.NONE},
        },
        "full_scene_id": DataType.DISCRETE,
        "compressed_scene_id": DataType.DISCRETE,
    }

    dataloader = dict(
        datasets=dict(
            bridge=dict(
                path=source_path,
                train_split="train",
                val_split="val",
                transform=ModuleSpec.create(bridge_dataset_transform),
            ),
        ),
        n_obs=2,
        n_action=1,
        n_action_bins=128,
        augment_kwargs=dict(
            scale_range=(0.8, 1.0),
            aspect_ratio_range=(1.16, 1.328),
            aligned=True,
        ),
        chunk_img=True,
        goal_conditioned=True,
        shuffle_size=200000,
        batch_size=256,
        recompute_statistics=False,
    )

    model = ModuleSpec.create(
        Model,
        encoders={
            "observation->image->agent,goal->image->agent": ModuleSpec.create(
                resnet_class, spatial_coordinates=True, act="swish", num_kp=None
            ),
            **(
                {
                    "observation->state": None,
                }
                if config_str == "state"
                else {}
            ),
        },
        trunk=ModuleSpec.create(Concatenate, features=None, flatten_time=True),
        action_head=ModuleSpec.create(
            DiscreteActionHead,
            model=ModuleSpec.create(
                MLP, hidden_dims=(512, 512, 512), dropout_rate=None, activate_final=True, use_layer_norm=True
            ),
            n_action_bins=128,
        ),
    )

    lr_schedule = ModuleSpec.create(
        optax.warmup_cosine_decay_schedule,
        init_value=1e-6,
        peak_value=3e-4,
        warmup_steps=1000,
        decay_steps=500000,
        end_value=1e-6,
    )
    optimizer = ModuleSpec.create(optax.adamw)

    envs = None
    return ConfigDict(
        dict(
            structure=structure,
            envs=envs,
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            # Add training parameters
            steps=500000,
            log_freq=500,
            val_freq=5000,
            eval_freq=20000,
            save_freq=100000,
            val_steps=25,
            seed=0,
            # Add doremi parameters
            smoothing=5e-3,
            domain_weight_step_size=1.0,
            domain_key="full_scene_id",
            num_domains=32,
            initial_alpha=[FULL_BRIDGE_WEIGHTS[i] for i in range(32)],
        )
    )
