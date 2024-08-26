"""
Flax implementation of ResNet V1.5.

Much of this implementation was borrowed from
https://github.com/google/flax/commits/main/examples/imagenet/models.py
under the APACHE 2.0 license. See the Flax repo for details.
"""

from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

ModuleDef = Any


class ResNetBlock(nn.Module):
    """ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(
        self,
        x,
    ):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm(self.filters // 16)(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm(self.filters // 16, scale_init=nn.initializers.zeros_init())(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides, name="conv_proj")(residual)
            residual = self.norm(self.filters // 16, name="norm_proj")(residual)

        return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros_init())(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters * 4, (1, 1), self.strides, name="conv_proj")(residual)
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)


class SpatialSoftmax(nn.Module):
    num_kp: Optional[int] = 64
    temperature: float = 1.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        # Input is shape (Batch dims...., H, W, C)
        if self.num_kp is not None:
            x = nn.Conv(self.num_kp, kernel_size=(1, 1), strides=1, name="keypoints")(x)

        h, w, c = x.shape[-3:]
        pos_x, pos_y = jnp.meshgrid(
            jnp.linspace(-1.0, 1.0, w, dtype=self.dtype), jnp.linspace(-1.0, 1.0, h, dtype=self.dtype)
        )
        pos_x, pos_y = pos_x.reshape((h * w, 1)), pos_y.reshape((h * w, 1))  # (H*W, 1)
        x = x.reshape(x.shape[:-3] + (h * w, c))  # (..., H, W, C)

        attention = jax.nn.softmax(x / self.temperature, axis=-2)  # (B..., H*W, K)
        expected_x = (pos_x * attention).sum(axis=-2, keepdims=True)  # (B..., 1, K)
        expected_y = (pos_y * attention).sum(axis=-2, keepdims=True)
        expected_xy = jnp.concatenate((expected_x, expected_y), axis=-2)  # (B..., 2, K)

        return expected_xy.reshape(x.shape[:-2] + (2 * c,))


class SpatialCoordinates(nn.Module):
    """
    Inspired by https://github.com/rail-berkeley/bridge_data_v2/blob/main/jaxrl_m/vision/resnet_v1.py
    but simplified to be a bit more readable and stay in jnp
    """

    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        h, w = x.shape[-3:-1]
        pos_x, pos_y = jnp.meshgrid(
            jnp.linspace(-1.0, 1.0, w, dtype=self.dtype), jnp.linspace(-1.0, 1.0, h, dtype=self.dtype)
        )
        coords = jnp.stack((pos_x, pos_y), axis=-1)  # (H, W, 2)
        coords = jnp.broadcast_to(coords, x.shape[:-3] + coords.shape)
        return jnp.concatenate((x, coords), axis=-1)


class ResNet(nn.Module):
    """ResNetV1.5, except with group norm instead of BatchNorm"""

    stage_sizes: Sequence[int] = (3, 4, 6, 3)
    block_cls: ModuleDef = ResNetBlock
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: str = "relu"
    conv: ModuleDef = nn.Conv
    spatial_coordinates: bool = False
    num_kp: Optional[int] = None

    @nn.compact
    def __call__(self, obs, goal: Optional[jax.Array] = None, train: bool = True):
        # Initialize layers
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            nn.GroupNorm,
            epsilon=1e-5,
            dtype=self.dtype,
        )
        act = getattr(jax.nn, self.act)

        # Determine whether or not we concatenate
        if goal is None:
            x = obs
        else:
            # Obs is shape (B, T, H, W, C), Goal is shape (B, 1, H, W, C)
            x = jnp.concatenate((obs, jnp.broadcast_to(goal, obs.shape)), axis=-1)  # Concat on channel axis

        # Shift inputs to -1 to 1 from 0 to 1
        x = 2 * x - 1

        if self.spatial_coordinates:
            # Add spatial coordinates.
            x = SpatialCoordinates(dtype=self.dtype)(x)

        x = conv(
            self.num_filters,
            (7, 7),
            (2, 2),
            padding=[(3, 3), (3, 3)],
            name="conv_init",
        )(x)
        x = norm(self.num_filters // 16, name="gn_init")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(
                    self.num_filters * 2**i,
                    strides=strides,
                    conv=conv,
                    norm=norm,
                    act=act,
                )(x)

        if self.num_kp is not None:
            return SpatialSoftmax(num_kp=self.num_kp)(x)
        else:
            # Perform average pooling over the enbmeddings.
            return jnp.mean(x, axis=(-3, -2))  # (..., H, W, C) -> (B, T, C).


class ResNet18(ResNet):
    stage_sizes: Sequence[int] = (2, 2, 2, 2)
    block_cls: ModuleDef = ResNetBlock
    num_kp: Optional[int] = 64


class ResNet34(ResNet):
    stage_sizes: Sequence[int] = (3, 4, 6, 3)
    block_cls: ModuleDef = ResNetBlock
    num_kp: Optional[int] = 96


class ResNet50(ResNet):
    stage_sizes: Sequence[int] = (3, 4, 6, 3)
    block_cls: ModuleDef = BottleneckResNetBlock
    num_kp: Optional[int] = 128
