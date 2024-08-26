# Define the config for robomimic
import os

import optax
from ml_collections import ConfigDict

from openx.data.datasets.bridge import bridge_dataset_transform
from openx.data.datasets.franka import franka_dataset_transform
from openx.data.datasets.oxe import toto_dataset_transform
from openx.data.mixes import (
    OXE_ALL,
    RTX_DOREMI_150K,
    RTX_DOREMI_200K,
    RTX_MIX,
    RTX_MIX_UNIFORM,
)
from openx.data.utils import NormalizationType, StateEncoding
from openx.networks.action_heads import DDPMActionHead
from openx.networks.core import Model
from openx.networks.mlp import Concatenate, MLPResNet
from openx.networks.resnet import ResNet50
from openx.networks.vit import SmallStem, ViT_S
from openx.utils.schedules import warmup_rsqrt_schedule
from openx.utils.spec import ModuleSpec


def get_config(config_str: str = "rtx,gaussian,100,400000"):
    data_mix, normalization, percent, train_steps = config_str.split(",")
    assert data_mix in {
        "all",
        "rtx",
        "rtx_uniform",
        "rtx_doremi_150k",
        "rtx_doremi_200k",
    }
    assert normalization in {"gaussian", "bounds"}
    assert percent.isdigit()
    assert train_steps.isdigit()
    normalization = dict(gaussian=NormalizationType.GAUSSIAN, bounds=NormalizationType.BOUNDS_5STDV)[normalization]
    percent = int(percent)
    train_steps = int(train_steps)


    structure = {
        "observation": {
            "image": {
                "agent": (224, 224),  # Height x width
            },
        },
        "action": {
            "desired_delta": {
                StateEncoding.EE_POS: normalization,
                StateEncoding.EE_EULER: normalization,
            },
            # Re-scale to -1 to 1 from 0 to 1 to emphasize gripper loss.
            "desired_absolute": {StateEncoding.GRIPPER: NormalizationType.NONE},
        },
    }

    # Get the dataset mix
    datasets = {
        "rtx": RTX_MIX,
        "rtx_uniform": RTX_MIX_UNIFORM,
        "rtx_doremi_150k": RTX_DOREMI_150K,
        "rtx_doremi_200k": RTX_DOREMI_200K,
    }[data_mix]

    # Get the subset sizes
    new_total_size = (percent / 100) * sum([OXE_ALL[dataset]["weight"] for dataset in datasets.keys()])
    total_weight = sum([dataset_config["weight"] for dataset_config in datasets.values()])
    # New sizes are the correct weights, subject to max data constraint
    new_sizes = {k: min(new_total_size * v["weight"] / total_weight, OXE_ALL[k]["weight"]) for k, v in datasets.items()}

    extra_data_points = new_total_size - sum(v for v in new_sizes.values())  # We might not fill all data points
    while extra_data_points > 0:  # We still have data to allocate
        # Get all datasets we have not filled.
        to_add = {k: OXE_ALL[k]["weight"] - v for k, v in new_sizes.items() if v < OXE_ALL[k]["weight"]}
        amount_to_add = min(extra_data_points / len(to_add), min(to_add.values()))
        # Add data from other datasets to fufill weighting
        for dataset in to_add:
            new_sizes[dataset] += amount_to_add
        extra_data_points = new_total_size - sum(v for v in new_sizes.values())  # We might not fill all data point

    percents = {k: 100 * v / OXE_ALL[k]["weight"] for k, v in new_sizes.items()}

    # Add the path to all the datasets
    # Allocate the parallel threads
    for dataset in datasets:
        datasets[dataset]["path"] = os.path.join(
            "PATH TO OCTO DATASETS", datasets[dataset]["path"]
        )
        train_split = datasets[dataset]["train_split"]
        assert train_split in {"train", "train[:95%]"}
        percent = 0.95 * percents[dataset] if train_split == "train[:95%]" else percents[dataset]
        percent = int(round(percent))
        datasets[dataset]["train_split"] = "train[:" + str(percent) + "%]"
        datasets[dataset]["num_parallel_reads"] = max(1, int(32 * datasets[dataset]["weight"] / total_weight))
        datasets[dataset]["num_parallel_calls"] = max(1, int(32 * datasets[dataset]["weight"] / total_weight))
        datasets[dataset]["weight"] = (
            0.92 * datasets[dataset]["weight"] / total_weight
        )  # make room for co-training split

    # Add in the cotraining datasets
    datasets["cotrain_bridge"] = dict(
        path="PATH TO COTRAINING BRIDGE DATASET",
        dataset_statistics="PATH TO BRIDGE DATASET (same statistics)",
        train_split="train",
        val_split="val",
        transform=ModuleSpec.create(bridge_dataset_transform),
        weight=0.04,
        num_parallel_reads=1,
        num_parallel_calls=1,
    )
    datasets["cotrain_franka"] = dict(
        path="PATH TO FRANKA DATASET",
        train_split="train",
        val_split="val",
        transform=ModuleSpec.create(franka_dataset_transform),
        weight=0.04,
        num_parallel_reads=1,
        num_parallel_calls=1,
    )

    dataloader = dict(
        datasets=datasets,
        n_obs=2,
        n_action=4,
        augment_kwargs=dict(
            scale_range=(0.8, 1.0),
            aspect_ratio_range=(0.9, 1.1),
            aligned=True,
            brightness=0.1,
            contrast_range=[0.9, 1.1],
            saturation_range=[0.9, 1.1],
            hue=0.05,
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

    
    encoder = ModuleSpec.create(ResNet50, spatial_coordinates=True, act="swish", num_kp=None)
    trunk = ModuleSpec.create(Concatenate, features=None, flatten_time=True)
    lr_schedule = ModuleSpec.create(
        optax.warmup_cosine_decay_schedule,
        init_value=1e-6,
        peak_value=2e-4,
        warmup_steps=1000,
        decay_steps=train_steps,
        end_value=1e-6,
    )
    optimizer = ModuleSpec.create(optax.adamw, mu_dtype="bfloat16")

    model = ModuleSpec.create(
        Model,
        encoders={
            "observation->image->agent,goal->image->agent": encoder,
        },
        trunk=trunk,
        action_head=ModuleSpec.create(
            DDPMActionHead,
            model=ModuleSpec.create(
                MLPResNet,
                hidden_dim=256,
                num_blocks=3,
                time_features=32,
                dropout_rate=None,
                use_layer_norm=True,
                learn_time_embedding=True,
            ),
            clip_sample=5.0 if normalization == NormalizationType.GAUSSIAN else 1.0,
            timesteps=50,
            variance_type="fixed_small",
        ),
    )

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
            clip_gradient=None,
            steps=train_steps,
            log_freq=500,
            val_freq=10000,
            eval_freq=20000,
            save_freq=50000,
            val_steps=20,
            seed=0,
        )
    )
