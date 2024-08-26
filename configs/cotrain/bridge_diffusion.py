import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import optax
from ml_collections import ConfigDict

from openx.data.datasets.bridge import bridge_dataset_transform
from openx.data.utils import DataType, NormalizationType, StateEncoding
from openx.networks.action_heads import DDPMActionHead
from openx.networks.core import Model
from openx.networks.mlp import Concatenate, MLPResNet
from openx.networks.resnet import ResNet50
from openx.utils.spec import ModuleSpec


def get_config(config_str: str = "state,True,50"):
    # By default returns the bridge config.
    data_type, augs, percent = config_str.split(",")
    assert data_type in {"state", "img"}
    assert augs in {"True", "False"}
    assert percent.isdigit()

    percent = int(percent)

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
                path="PATH TO BRIDGE DATASET",
                train_split="train[:" + str(percent) + "%]",
                val_split="val",
                transform=ModuleSpec.create(bridge_dataset_transform),
                weight=0.95,
            ),
            cotrain=dict(
                path="PATH TO COTRAIN DATASET",
                dataset_statistics="PATH TO BRIDGE DATASET (Same statistics)",
                train_split="train",
                val_split="val",
                transform=ModuleSpec.create(bridge_dataset_transform),
                weight=0.05,
            ),
        ),
        n_obs=2,
        n_action=2,
        augment_kwargs=dict(
            scale_range=(0.8, 1.0),
            aspect_ratio_range=(1.12, 1.328),
            aligned=True,
            **(
                dict(
                    brightness=0.1,
                    contrast_range=[0.9, 1.1],
                    saturation_range=[0.9, 1.1],
                    hue=0.025,
                )
                if augs == "True"
                else dict()
            ),
        ),
        chunk_img=True,
        goal_conditioned=True,
        shuffle_size=200000,
        batch_size=384,
        recompute_statistics=False,
    )

    model = ModuleSpec.create(
        Model,
        encoders={
            "observation->image->agent,goal->image->agent": ModuleSpec.create(
                ResNet50,
                spatial_coordinates=True,
                act="swish",
                num_kp=None,
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
            DDPMActionHead,
            model=ModuleSpec.create(
                MLPResNet, hidden_dim=256, num_blocks=3, time_features=64, dropout_rate=None, use_layer_norm=True
            ),
            clip_sample=1.0,
            timesteps=100,
            variance_type="fixed_small",
        ),
    )

    lr_schedule = ModuleSpec.create(
        optax.warmup_cosine_decay_schedule,
        init_value=1e-6,
        peak_value=2e-4,
        warmup_steps=1000,
        decay_steps=500000,
        end_value=1e-6,
    )
    optimizer = ModuleSpec.create(optax.adamw)

    envs = None
    return ConfigDict(
        dict(
            structure=structure,
            domain_key="full_scene_id",
            num_domains=32,
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
        )
    )
