"""
ViT Implementation
Borrows from Google Big Vision and Octo
"""

from typing import Any, Callable, Dict, Optional, TypeVar

import flax.linen as nn
import jax
import jax.numpy as jnp

T = TypeVar("T")


class PatchEncoder(nn.Module):
    embed_dim: int
    patch_size: int = 16
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, obs, goal: Optional[jax.Array], train: bool = True):
        # Determine whether or not we concatenate
        if goal is None:
            x = obs
        else:
            # Obs is shape (B, T, H, W, C), Goal is shape (B, 1, H, W, C)
            x = jnp.concatenate((obs, jnp.broadcast_to(goal, obs.shape)), axis=-1)  # Concat on channel axis

        # Shift inputs to -1 to 1 from 0 to 1
        x = 2 * x - 1

        x = nn.Conv(
            self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            name="embedding",
            dtype=self.dtype,
        )(x)

        B, T, H, W, C = x.shape
        x = jnp.reshape(x, [B, T, H * W, C])
        return x


def weight_standardize(w, axis, eps: float = 1e-5):
    """Subtracts mean and divides by standard deviation."""
    w = w - jnp.mean(w, axis=axis)
    w = w / (jnp.std(w, axis=axis) + eps)
    return w


class StdConv(nn.Conv):
    """Convolution with weight standardization."""

    def param(self, name: str, init_fn: Callable[..., T], *init_args) -> T:
        param = super().param(name, init_fn, *init_args)
        if name == "kernel":
            param = weight_standardize(param, axis=[0, 1, 2], eps=1e-5)
        return param


class SmallStem(nn.Module):
    """
    Passes the image through a few light-weight convolutional layers,
    before patchifying the image. Empirically useful for many computer vision tasks.

    See Xiao et al: Early Convolutions Help Transformers See Better
    """

    embed_dim: int
    patch_size: int = 16
    kernel_sizes: tuple = (3, 3, 3, 3)
    strides: tuple = (2, 2, 2, 2)
    features: tuple = (32, 96, 192, 384)  # modified from 48 -> 32 first layer for GroupNorm
    padding: tuple = (1, 1, 1, 1)
    use_std_conv: bool = True
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, obs, goal: Optional[jax.Array], train: bool = True):
        # Determine whether or not we concatenate
        if goal is None:
            x = obs
        else:
            # Obs is shape (B, T, H, W, C), Goal is shape (B, 1, H, W, C)
            x = jnp.concatenate((obs, jnp.broadcast_to(goal, obs.shape)), axis=-1)  # Concat on channel axis

        # Shift inputs to -1 to 1 from 0 to 1
        x = 2 * x - 1

        conv_cls = StdConv if self.use_std_conv else nn.Conv
        for kernel_size, stride, features, padding in zip(
            self.kernel_sizes,
            self.strides,
            self.features,
            self.padding,
            strict=False,
        ):
            x = conv_cls(
                features=features,
                kernel_size=(kernel_size, kernel_size),
                strides=(stride, stride),
                padding=padding,
            )(x)
            x = nn.GroupNorm(epsilon=1e-5, dtype=self.dtype)(x)
            x = nn.relu(x)

        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size // 16, self.patch_size // 16),
            strides=(self.patch_size // 16, self.patch_size // 16),
            padding="VALID",
            name="embedding",
        )(x)

        B, T, H, W, C = x.shape
        x = jnp.reshape(x, [B, T, H * W, C])  # (B, T, Num Tokens, C)
        return x


class PositionalEmbedding(nn.Module):
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        T, D = x.shape[-2:]  # Get (T, D)
        emb = self.param("positional_embedding", nn.initializers.normal(stddev=0.02), (1, T, D), self.dtype)
        return x + emb


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = True):
        """Applies Transformer MlpBlock module."""
        inits = dict(
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6),
        )
        D = x.shape[-1]
        x = nn.Dense(self.mlp_dim or 4 * D, dtype=self.dtype, **inits)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = nn.Dense(D, dtype=self.dtype, **inits)(x)
        return x


class EncoderBlock(nn.Module):
    """Single transformer encoder block (MHSA + MLP)."""

    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    dtype: Any = jnp.float32
    broadcast_dropout: bool = False

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Attention Block
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=not train,
            dropout_rate=self.attention_dropout_rate,
            dtype=self.dtype,
        )(y, y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
        x = x + y

        # MLP Block
        y = nn.LayerNorm()(x)
        y = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
        )(y, train=train)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
        x = x + y
        return x


class TransformerEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    num_layers: int
    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = True):
        for i in range(self.num_layers):
            x = EncoderBlock(
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                name=f"encoder_block_{i}",
                num_heads=self.num_heads,
                dtype=self.dtype,
            )(x, train=train)
        return nn.LayerNorm(name="encoder_norm")(x)


class MAPHead(nn.Module):
    """Multihead Attention Pooling."""

    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12

    @nn.compact
    def __call__(self, x, train: bool = True):
        # TODO
        B, L, D = x.shape  # pylint: disable=unused-variable
        probe = self.param("probe", nn.initializers.xavier_uniform(), (1, 1, D), x.dtype)
        probe = jnp.tile(probe, [B, 1, 1])

        x = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, kernel_init=nn.initializers.xavier_uniform())(
            probe, x
        )

        # TODO: dropout on head?
        y = nn.LayerNorm()(x)
        x = x + MlpBlock(mlp_dim=self.mlp_dim)(y, train=train)
        return x[:, 0]


class ViT(nn.Module):
    embed_dim: int
    num_layers: int
    num_heads: int
    num_registers: Optional[int] = None
    mlp_dim: Optional[int] = None
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.0
    pool_type: str = "cls"
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, modalities: Dict[str, jax.Array], train: bool = True):
        # Each Modality should be encoded as (B, T, Tokens, D). Thus, concatenate modalities on the Token dim.
        for modality in modalities.values():
            assert (
                modality.shape[-1] == self.embed_dim
            ), f"Modality {modality} did not end with embedding dim {self.embed_dim}"

        x = jnp.concatenate([modalities[k] for k in sorted(modalities.keys())], axis=-2)
        # Then, flatten the time dimension
        B, T, N, _ = x.shape  # (Batch, Time, Num Tokens, Dim)
        x = jnp.reshape(x, (B, T * N, self.embed_dim))

        # Add positional embedding
        x = PositionalEmbedding(dtype=self.dtype)(x)
        # Add registers
        if self.num_registers is not None:
            registers = self.param(
                "registers",
                nn.initializers.normal(stddev=0.02),
                (1, self.num_registers, self.embed_dim),
                self.dtype,
            )
            x = jnp.concatenate((x, jnp.tile(registers, [B, 1, 1])), axis=1)  # Registers at the end
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)  # Don't dropout CLS token.

        if self.pool_type == "cls":
            cls_token = self.param("cls", nn.initializers.zeros, (1, 1, self.embed_dim), x.dtype)
            x = jnp.concatenate((jnp.tile(cls_token, [B, 1, 1]), x), axis=1)  # CLS at the beginning

        # Run the transformer
        x = TransformerEncoder(
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            dtype=self.dtype,
        )(x, train=train)

        # Collect outputs via specified pooling
        if self.pool_type == "cls":
            x = x[:, 0]  # (B, D)
        elif self.pool_type == "avg":
            x = jnp.mean(x[:, : T * N], axis=1)  # Ignore registers
        elif self.pool_type == "map":
            x = MAPHead(num_heads=self.num_heads, mlp_dim=self.mlp_dim)(x[:, : T * N])  # Ignore registers
        else:
            raise ValueError(f"Unknown pool type: '{self.pool_type}'")

        return x


class ViT_T(ViT):
    embed_dim: int = 192
    num_layers: int = 12
    mlp_dim: int = 768
    num_heads: int = 3
    num_registers: int = 4
    dropout_rate: float = 0.0


class ViT_S(ViT):
    embed_dim: int = 384
    num_layers: int = 12
    mlp_dim: int = 1536
    num_heads: int = 6
    num_registers: int = 4
    dropout_rate: float = 0.0


class ViT_B(ViT):
    embed_dim: int = 768
    num_layers: int = 12
    mlp_dim: int = 3072
    num_heads: int = 12
    num_registers: int = 8
    dropout_rate: float = 0.0
