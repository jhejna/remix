# Define the config for robomimic
import os

import optax
from ml_collections import ConfigDict

from openx.data.mixes import OXE_ALL, OXE_MAGIC_SOUP_SUBSET, RTX_MIX_UNIFORM
from openx.data.utils import NormalizationType, StateEncoding
from openx.networks.action_heads import DiscreteActionHead
from openx.networks.core import Model
from openx.networks.mlp import MLP, Concatenate
from openx.networks.resnet import ResNet18, ResNet34, ResNet50
from openx.utils.spec import ModuleSpec


def get_config(config_str: str = "soup,size,50"):
    data_mix, data_weight, resnet_class = config_str.split(",")
    assert data_mix in {"all", "magic_soup", "rtx"}
    assert data_weight in {"size", "uniform"}
    assert resnet_class in {"18", "34", "50"}

    resnet_class = {"18": ResNet18, "34": ResNet34, "50": ResNet50}[resnet_class]

    structure = {
        "observation": {
            "image": {
                "agent": (224, 224),  # Height x width
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
    datasets = {"all": OXE_ALL, "magic_soup": OXE_MAGIC_SOUP_SUBSET, "rtx": RTX_MIX_UNIFORM}[data_mix]

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
            "PATH TO OCTO DATASET", datasets[dataset]["path"]
        )
        # datasets[dataset]["val_split"] = None  # Remove validation splits -- we don't need them.
        datasets[dataset]["num_parallel_reads"] = max(1, int(32 * datasets[dataset]["weight"] / total_weight))
        datasets[dataset]["num_parallel_calls"] = max(1, int(32 * datasets[dataset]["weight"] / total_weight))
        datasets[dataset]["weight"] = max(datasets[dataset]["weight"] / total_weight, 0.01)  # Everything is at least 1%

    dataloader = dict(
        datasets=datasets,
        n_obs=2,
        n_action=1,
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
        batch_size=512,
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
                resnet_class, spatial_coordinates=True, act="swish", num_kp=None
            ),
        },
        trunk=ModuleSpec.create(Concatenate, features=None, flatten_time=True),
        action_head=ModuleSpec.create(
            DiscreteActionHead,
            model=ModuleSpec.create(
                MLP, hidden_dims=(512, 512, 512), dropout_rate=None, activate_final=True, use_layer_norm=True
            ),
            n_action_bins=100,
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
            steps=500000,
            log_freq=250,
            val_freq=5000,
            eval_freq=20000,
            save_freq=50000,
            val_steps=25,
            seed=0,
            # Add doremi parameters
            domain_weight_step_size=0.2,
            domain_key="dataset_id",
            smoothing=5e-2,
        )
    )
