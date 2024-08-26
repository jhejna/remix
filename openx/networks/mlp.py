from typing import Callable, Dict, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

default_init = nn.initializers.xavier_uniform


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                # In the case of using layernorm and dropout, prefer doing it this way for the actor
                # It doesn't make sense to do dropout -> layernorm because it messes up statistics at test time.
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
                x = self.activation(x)
        return x


class Concatenate(nn.Module):
    features: Optional[int] = None
    flatten_time: bool = True

    @nn.compact
    def __call__(self, modalities: Dict[str, jax.Array], train: bool = False):
        x = jnp.concatenate([modalities[k] for k in sorted(modalities.keys())], axis=-1)  # (B, T, D)
        if self.features is not None:
            x = nn.Dense(self.features)(x)
        if self.flatten_time:
            x = x.reshape((x.shape[0], -1))
        return x


class MLPResNetBlock(nn.Module):
    features: int
    act: Callable
    dropout_rate: float = None
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x, train: bool = False):
        residual = x
        if self.dropout_rate is not None and self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.Dense(self.features * 4)(x)
        x = self.act(x)
        x = nn.Dense(self.features)(x)

        if residual.shape != x.shape:
            residual = nn.Dense(self.features)(residual)

        return residual + x


class SinusoidalPosEmb(nn.Module):
    features: int
    learned: bool = False

    @nn.compact
    def __call__(self, x: jax.Array):
        half_features = self.features // 2
        if self.learned:
            w = self.param("kernel", nn.initializers.normal(0.2), (half_features, x.shape[-1]), jnp.float32)
            emb = 2 * jnp.pi * x @ w.T
        else:
            emb = jnp.log(10000) / (half_features - 1)
            emb = jnp.exp(jnp.arange(half_features) * -emb)
            emb = x * emb
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb


class MLPResNet(nn.Module):
    num_blocks: int
    dropout_rate: float = None
    use_layer_norm: bool = False
    hidden_dim: int = 256
    activation: Callable = nn.swish
    time_features: int = 64
    learn_time_embedding: bool = False

    @nn.compact
    def __call__(self, obs, action, time, train: bool = False) -> jax.Array:
        time = SinusoidalPosEmb(self.time_features, learned=self.learn_time_embedding)(time)
        time = MLP(hidden_dims=(2 * self.time_features, self.time_features), activation=nn.swish, activate_final=False)(
            time, train=train
        )
        # Obs is (B, D). Action is (B, H, D), time is (B, D)
        H, action_dim = action.shape[-2:]
        action = action.reshape(-1, H * action_dim)
        x = jnp.concatenate((obs, action, time), axis=-1)
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        for _ in range(self.num_blocks):
            x = MLPResNetBlock(
                self.hidden_dim,
                act=self.activation,
                use_layer_norm=self.use_layer_norm,
                dropout_rate=self.dropout_rate,
            )(x, train=train)
        x = self.activation(x)  # Shape (B, hidden_dim)
        return x
