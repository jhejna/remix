# Define the config for robomimic
import os

import optax
from ml_collections import ConfigDict

from openx.data.mixes import OXE_ALL, OXE_MAGIC_SOUP, RTX_MIX
from openx.data.utils import NormalizationType, StateEncoding
from openx.networks.action_heads import DDPMActionHead
from openx.networks.core import Model
from openx.networks.mlp import Concatenate, MLPResNet
from openx.networks.resnet import ResNet50
from openx.utils.spec import ModuleSpec


def get_config(config_str: str = "all,size"):
    data_mix, data_weight = config_str.split(",")
    assert data_mix in {"all", "magic_soup", "rtx"}
    assert data_weight in {"size", "uniform"}

    structure = {
        "observation": {
            "image": {
                "agent": (224, 224),  # Height x width
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
    datasets = {"all": OXE_ALL, "magic_soup": OXE_MAGIC_SOUP, "rtx": RTX_MIX}[data_mix]

    if data_weight == "uniform":
        # Weight uniformly
        for dataset in datasets:
            datasets[dataset]["weight"] = 1.0
    else:
        assert all([dataset_config["weight"] != 1.0 for dataset_config in datasets.values()])

    total_weight = sum([dataset_config["weight"] for dataset_config in datasets.values()])

    # Add the path to all the datasets
    # Allocate the parallel threads
    for dataset in datasets:
        datasets[dataset]["path"] = os.path.join(
            "PATH TO OCTO DATASETS", datasets[dataset]["path"]
        )
        datasets[dataset]["num_parallel_reads"] = max(1, int(32 * datasets[dataset]["weight"] / total_weight))
        datasets[dataset]["num_parallel_calls"] = max(1, int(32 * datasets[dataset]["weight"] / total_weight))

    dataloader = dict(
        datasets=datasets,
        n_obs=2,
        n_action=2,
        augment_kwargs=dict(
            scale_range=(0.85, 1.0),
            aspect_ratio_range=(0.85, 1.15),
            aligned=True,
            brightness=0.1,
            contrast_range=[0.9, 1.1],
            saturation_range=[0.9, 1.1],
            hue=0.03,
        ),
        chunk_img=True,
        goal_conditioned=True,
        shuffle_size=500000,
        batch_size=384,
        recompute_statistics=False,
        weight_by_size=False,
        num_parallel_calls=128,
        num_batch_parallel_calls=None,
        restrict_memory=True,
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
            envs=envs,
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            # Add training parameters
            steps=400000,
            log_freq=500,
            val_freq=10000,
            eval_freq=20000,
            save_freq=50000,
            val_steps=20,
            seed=0,
        )
    )
