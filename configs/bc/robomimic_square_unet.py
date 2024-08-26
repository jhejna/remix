# Define the config for robomimic
import os

import optax
from ml_collections import ConfigDict

from openx.data.utils import NormalizationType, StateEncoding
from openx.networks.action_heads import DDPMActionHead
from openx.networks.core import Model
from openx.networks.mlp import Concatenate
from openx.networks.unet import ConditionalUnet1D
from openx.networks.resnet import ResNet18
from openx.utils.spec import ModuleSpec
from openx.data.datasets.robomimic import robomimic_dataset_transform
from openx.envs.robomimic import RobomimicEnv

DOMAIN_SIZES = {
        'better_operator_1': 6948, 
        'better_operator_2': 9919, 
        'okay_operator_1': 13571, 
        'okay_operator_2': 10474, 
        'worse_operator_1': 14548, 
        'worse_operator_2': 17247
    }

DOREMI_WEIGHTS = {
        "better_operator_1": 0.22780823707580566,
        "better_operator_2": 0.19992227852344513,
        "okay_operator_1": 0.11927580088376999,
        "okay_operator_2": 0.1463860124349594,
        "worse_operator_1": 0.17970919609069824,
        "worse_operator_2": 0.126898854970932
    }

UNIFORM_WEIGHTS = {k: v / sum(DOMAIN_SIZES.values()) for k, v in DOMAIN_SIZES.items()}

def get_config(config_str: str = "100,doremi"):
    percent, weights = config_str.split(",")
    assert weights in {"doremi", "uniform"}
    assert percent.isnumeric()
    percent = float(percent)

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
                StateEncoding.EE_POS: NormalizationType.BOUNDS,
                StateEncoding.EE_EULER: NormalizationType.BOUNDS,
            },
            "desired_absolute": {StateEncoding.GRIPPER: NormalizationType.NONE},
        },
    }

    # Get the dataset mix
    weights = DOREMI_WEIGHTS if weights == "doremi" else UNIFORM_WEIGHTS
   
    # Get the subset sizes
    new_total_size = (percent / 100) * sum(DOMAIN_SIZES.values())
    total_weight = sum(weights.values())
    # New sizes are the correct weights, subject to max data constraint
    new_sizes = {k: min(new_total_size * v / total_weight, DOMAIN_SIZES[k]) for k, v in weights.items()}

    extra_data_points = new_total_size - sum(v for v in new_sizes.values())  # We might not fill all data points
    while extra_data_points > 0:  # We still have data to allocate
        # Get all datasets we have not filled.
        to_add = {k: DOMAIN_SIZES[k] - v for k, v in new_sizes.items() if v < DOMAIN_SIZES[k]}
        amount_to_add = min(extra_data_points / len(to_add), min(to_add.values()))
        # Add data from other datasets to fufill weighting
        for dataset in to_add:
            new_sizes[dataset] += amount_to_add
        extra_data_points = new_total_size - sum(v for v in new_sizes.values())  # We might not fill all data point

    percents = {k: 100 * v / DOMAIN_SIZES[k] for k, v in new_sizes.items()}

    datasets = {}
    for name, percent in percents.items():
        datasets[name] = {
            "path": "PATH TO CONVERTED ROBOMIMIC DATASETS",
            "train_split": name + "_train" + "[:" + str(percent) + "%]",
            "val_split": name + "_valid",
            "transform": ModuleSpec.create(robomimic_dataset_transform),
            "weight": weights[name]
        }


    dataloader = dict(
        datasets=datasets,
        n_obs=2,
        n_action=16,
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
        trunk=ModuleSpec.create(Concatenate, features=128, flatten_time=True),
        action_head=ModuleSpec.create(
            DDPMActionHead,
            model=ModuleSpec.create(
                ConditionalUnet1D, down_features=(256, 512, 1024), mid_layers=2, time_features=128, kernel_size=5
            ),
            clip_sample=1.0,
            timesteps=100,
            variance_type="fixed_small",
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

    envs = dict(better_operator_1=ModuleSpec.create(RobomimicEnv, path="PATH TO ROBOMIMIC HDF5", horizon=500))
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
            log_freq=200,
            val_freq=5000,
            eval_freq=25000,
            save_freq=25000,
            val_steps=20,
            n_eval_proc=24,
            eval_ep=24,
            seed=0,
        )
    )
