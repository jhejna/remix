# Define the config for robomimic
import os

import optax
from ml_collections import ConfigDict

from openx.data.mixes import OXE_ALL, OXE_MAGIC_SOUP_SUBSET, RTX_MIX_UNIFORM
from openx.data.utils import NormalizationType, StateEncoding
from openx.networks.action_heads import DiscreteActionHead
from openx.networks.core import Model
from openx.networks.mlp import MLP, Concatenate
from openx.networks.resnet import ResNet18
from openx.utils.spec import ModuleSpec
from openx.data.datasets.robomimic import robomimic_dataset_transform


def get_config(config_str: str = ""):

    structure = {
        "observation": {
            "state": {
                StateEncoding.EE_POS: NormalizationType.NONE,
                StateEncoding.EE_QUAT: NormalizationType.NONE,
                StateEncoding.GRIPPER: NormalizationType.NONE,
            },
            "image": {
                "agent": (84, 84),  # Height x width
                "wrist": (84, 84),
            },
        },
        "action": {
            "desired_delta": {
                StateEncoding.EE_POS: NormalizationType.GAUSSIAN,
                StateEncoding.EE_EULER: NormalizationType.GAUSSIAN,
            },
            "desired_absolute": {StateEncoding.GRIPPER: NormalizationType.NONE},
        },
    }

    # Get the dataset mix
    operators = ["better_operator_1", "better_operator_2", "okay_operator_1", "okay_operator_2", "worse_operator_1", "worse_operator_2"]
    datasets = {}
    for operator in operators:
        datasets[operator] = {
            "path": "PATH TO CONVERTED ROBOMIMIC DATASET",
            "train_split": operator + "_train",
            "val_split": operator + "_valid",
            "transform": ModuleSpec.create(robomimic_dataset_transform),
        }

    dataloader = dict(
        datasets=datasets,
        n_obs=2,
        n_action=1,
        augment_kwargs=dict(
            scale_range=(0.85, 1.0),
            aspect_ratio_range=(0.9, 1.1),
        ),
        chunk_img=True,
        goal_conditioned=False,
        shuffle_size=100000,
        batch_size=256,
        recompute_statistics=False,
        weight_by_size=False,
    )

    model = ModuleSpec.create(
        Model,
       encoders={
            "observation->image->agent": ModuleSpec.create(ResNet18),
            "observation->image->wrist": ModuleSpec.create(ResNet18),
            "observation->state": None,
        },
        trunk=ModuleSpec.create(Concatenate, features=None, flatten_time=True),
        action_head=ModuleSpec.create(
            DiscreteActionHead,
            model=ModuleSpec.create(
                MLP, hidden_dims=(512, 512, 512), dropout_rate=0.4, activate_final=True, use_layer_norm=True
            ),
            n_action_bins=48,
            bin_type="gaussian",
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
            steps=10000,
            log_freq=200,
            val_freq=1000,
            eval_freq=20000,
            save_freq=1000,
            val_steps=20,
            seed=0,
            # Add doremi parameters
            domain_weight_step_size=0.2,
            domain_key="dataset_id",
            smoothing=5e-2,
        )
    )
