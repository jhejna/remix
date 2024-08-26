import abc
from typing import Optional

import distrax
import jax
from flax import linen as nn
from jax import numpy as jnp


class ActionHead(nn.Module, abc.ABC):
    model: nn.Module
    action_dim: int
    action_horizon: int

    @abc.abstractmethod
    def predict(self, obs: jax.Array, train: bool = True):
        raise NotImplementedError

    @abc.abstractmethod
    def loss(self, obs: jax.Array, action: jax.Array, train: bool = True):
        raise NotImplementedError


class L2ActionHead(ActionHead):
    @nn.compact
    def __call__(self, obs: jax.Array, train: bool = True):
        # Assume observation passed in is of shape (B, T, D) or (B, D)
        x = self.model(obs, train=train)
        # Handles whether or not the model predicts time.
        pred_dim = self.action_dim if len(x.shape) == 3 else self.action_dim * self.action_horizon
        x = nn.Dense(pred_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        x = x.reshape((obs.shape[0], self.action_horizon, self.action_dim))
        return x

    def predict(self, obs: jax.Array, train: bool = True):
        return self(obs, train=train)

    def loss(self, obs: jax.Array, action: jax.Array, train: bool = True):
        pred = self(obs, train=train)
        return jnp.square(pred - action).sum(axis=-1)  # (B, T, D) --> (B, T)


class DiscreteActionHead(ActionHead):
    n_action_bins: int
    bin_type: str = "uniform"
    temperature: Optional[bool] = None

    def setup(self):
        assert self.n_action_bins <= 256, "Maximum action bins supported is 256 due to uint8."
        if self.bin_type == "uniform":
            self.bins = jnp.linspace(-1, 1, self.n_action_bins + 1)
        elif self.bin_type == "gaussian":
            # Values chosen to approximate -5 to 5
            self.bins = jax.scipy.stats.norm.ppf(jnp.linspace(5e-3, 1 - 5e-3, self.n_action_bins + 1), scale=2)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

    @nn.compact
    def __call__(self, obs: jax.Array, train: bool = True):
        x = self.model(obs, train=train)
        pred_dim = self.action_dim if len(x.shape) == 3 else self.action_dim * self.action_horizon
        x = nn.Dense(pred_dim * self.n_action_bins, kernel_init=nn.initializers.xavier_uniform())(x)
        x = x.reshape((obs.shape[0], self.action_horizon, self.action_dim, self.n_action_bins))
        return x

    def predict(self, obs: jax.Array, train: bool = True):
        logits = self(obs, train=train)
        # by default we are not going to sample
        if self.temperature is None:
            action = jnp.argmax(logits, axis=-1)
        else:
            rng, key = jax.random.split(self.make_rng("dropout"))
            dist = distrax.Categorical(logits=logits / self.temperature)
            action = dist.sample(seed=key).astype(jnp.int32)
        return self.bin_centers[action]

    def loss(self, obs: jax.Array, action: jax.Array, train: bool = True):
        logits = self(obs, train=train)  # (B, T, D, N)

        # Clip the actions to be in range
        if self.bin_type == "uniform":
            action = jnp.clip(action, -1, 1)
        else:
            action = jnp.clip(action, -5, 5)

        # Compute the binned actions
        action = action[..., None]  # (B, T, D, 1)
        action_one_hot = (action < self.bins[1:]) & (action >= self.bins[:-1])
        action_one_hot = action_one_hot.astype(logits.dtype)

        logprobs = jax.nn.log_softmax(logits, axis=-1)  # (B, T, D, N)
        return -jnp.sum(logprobs * action_one_hot, axis=(-1, -2))  # Sum over dist and action dims


def _squaredcos_cap_v2(timesteps, s=0.008):
    t = jnp.linspace(0, timesteps, timesteps + 1) / timesteps
    alphas_cumprod = jnp.cos((t + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0, 0.999)


class DDPMActionHead(ActionHead):
    """
    Diffusion action head. Based on the DDPM implementation from Octo and Bridge.
    """

    timesteps: int = 100
    clip_sample: Optional[float] = None
    variance_type: str = "fixed_large"

    def setup(self):
        self.action_proj = nn.Dense(self.action_dim)
        betas = _squaredcos_cap_v2(self.timesteps).astype(jnp.float32)
        self.alphas = 1.0 - betas  # So betas = 1 - alphas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)

    @nn.compact
    def __call__(
        self, obs: jax.Array, time: Optional[jax.Array] = None, action: Optional[jax.Array] = None, train: bool = True
    ):
        if self.is_initializing():
            time = jnp.zeros(shape=(1, 1), dtype=int)
            action = jnp.zeros((1, self.action_horizon, self.action_dim), dtype=jnp.float32)
        x = self.model(obs, action=action, time=time, train=train)
        # Handles whether or not the model predicts time.
        pred_dim = self.action_dim if len(x.shape) == 3 else self.action_dim * self.action_horizon
        x = nn.Dense(pred_dim)(x)
        x = x.reshape((obs.shape[0], self.action_horizon, self.action_dim))
        return x

    def loss(self, obs: jax.Array, action: jax.Array, train: bool = True):
        # handle rng creation
        time_key, noise_key = jax.random.split(self.make_rng("dropout"))
        time = jax.random.randint(time_key, shape=(action.shape[0], 1), minval=0, maxval=self.timesteps)  # (B, 1)
        noise = jax.random.normal(noise_key, action.shape)  # (B, T, D)

        # Add noise to the action according to the schedule
        sqrt_alpha_prod = jnp.sqrt(self.alphas_cumprod[time[:, None]])  # (B, 1, 1)
        sqrt_one_minus_alpha_prod = jnp.sqrt(1 - self.alphas_cumprod[time[:, None]])  # (B, 1, 1)
        if self.clip_sample is not None:
            # If we are clipping at inference time, better assume the same range for train time!
            action = jnp.clip(action, -self.clip_sample, self.clip_sample)
        noisy_action = sqrt_alpha_prod * action + sqrt_one_minus_alpha_prod * noise

        pred = self(obs, time=time, action=noisy_action, train=train)

        return jnp.square(pred - noise).sum(axis=-1)  # (B, T, D) --> (B, T)

    def predict(self, obs: jax.Array, train: bool = True):
        """
        Code inspired by diffusers:
        https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm_flax.py
        """
        module, variables = self.unbind()

        def loop_body(i, args):
            sample, rng = args
            time = self.timesteps - 1 - i
            # Note that here time is (B, 1, 1) where as in loss in is (B, 1)
            time = jnp.broadcast_to(time, (sample.shape[0], 1, 1))
            alpha = self.alphas[time]
            alpha_prod_t = self.alphas_cumprod[time]
            alpha_prod_t_prev = jnp.where(time > 0, self.alphas_cumprod[time - 1], jnp.array(1.0, dtype=jnp.float32))

            # Run the model. Reduce time to (B, 1) for the model.
            eps = module.apply(variables, obs, time=time[:, 0], action=sample, train=train)

            # Predict x_0, clip if desired.
            orig = (sample - jnp.sqrt(1 - alpha_prod_t) * eps) / jnp.sqrt(alpha_prod_t)
            if self.clip_sample is not None:
                orig = jnp.clip(orig, -self.clip_sample, self.clip_sample)

            # Compute x_{t-1} using x_0
            orig_coeff = jnp.sqrt(alpha_prod_t_prev) * (1 - alpha) / (1 - alpha_prod_t)
            current_coeff = jnp.sqrt(alpha) * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)

            prev = orig_coeff * orig + current_coeff * sample

            # Add noise according to the schedule
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha)
            if self.variance_type == "fixed_large":
                variance = 1 - alpha
            elif self.variance_type == "fixed_small":
                variance = jnp.clip(variance, a_min=1e-20)
            else:
                raise ValueError("Invalid schedule provided")

            rng, key = jax.random.split(rng)
            variance = jnp.where(time > 0, variance, jnp.zeros(eps.shape, dtype=jnp.float32))
            prev = prev + jnp.sqrt(variance) * jax.random.normal(key, shape=sample.shape, dtype=jnp.float32)
            return (prev, rng)

        rng, key = jax.random.split(self.make_rng("dropout"))
        noisy_action = jax.random.normal(key, (obs.shape[0], self.action_horizon, self.action_dim), dtype=jnp.float32)
        noisy_action, _ = jax.lax.fori_loop(0, self.timesteps, loop_body, (noisy_action, rng))

        return noisy_action
