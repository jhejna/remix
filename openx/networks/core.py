from typing import Dict

from flax import linen as nn
from jax import numpy as jnp

"""
Defines the core model
"""


class Model(nn.Module):
    encoders: Dict[str, nn.Module]
    trunk: nn.Module
    action_head: nn.Module

    def _encode(self, batch: Dict, train: bool = True):
        # Support passing multiple modalities to a single encoder. Specify via tuple-key with `->`
        modalities = dict()
        for encoder_keys, encoder in self.encoders.items():
            # Assemble the args for the different modules
            args = []
            for encoder_key in encoder_keys.split(","):
                v = batch
                for k in encoder_key.split("->"):
                    v = v[k]
                args.append(v)
            args = tuple(args)
            if encoder is None:
                modalities[encoder_keys] = args[0] if len(args) == 1 else args
            else:
                modalities[encoder_keys] = encoder(*args, train=train)
        return modalities

    def __call__(self, batch: Dict, train: bool = True):
        # Exists so we get full tracing with module.init
        # Should not be used for training.
        obs = self._encode(batch, train=train)
        obs = self.trunk(obs, train=train)
        if self.is_initializing():
            return self.action_head(obs, train=train)
        else:
            return obs  # Return the encoded observation, keep action head separate.

    def predict(self, batch: Dict, train: bool = True):
        x = self(batch, train=train)
        return self.action_head.predict(x, train=train)

    def loss(self, batch: Dict, reduce: bool = True, train: bool = True):
        x = self(batch, train=train)
        loss = self.action_head.loss(x, batch["action"], train=train)
        loss = loss * batch["mask"]
        if reduce:
            loss = jnp.mean(loss) / jnp.clip(jnp.mean(batch["mask"]), a_min=1e-5, a_max=None)
        else:
            loss = jnp.mean(loss, axis=-1)
        return loss

    def loss_and_prediction_mse(self, batch: Dict, reduce: bool = True, train: bool = True):
        x = self(batch, train=train)
        loss = self.action_head.loss(x, batch["action"], train=train)
        loss = loss * batch["mask"]
        pred = self.action_head.predict(x, train=train)
        mse = jnp.square(pred - batch["action"]).sum(axis=-1)  # (B, T, D) --> (B, T)
        mse = mse * batch["mask"]
        if reduce:
            loss = jnp.mean(loss) / jnp.clip(jnp.mean(batch["mask"]), a_min=1e-5, a_max=None)
            mse = jnp.mean(mse) / jnp.clip(jnp.mean(batch["mask"]), a_min=1e-5, a_max=None)
        else:
            loss = jnp.mean(loss, axis=-1)
            mse = jnp.mean(mse, axis=-1)
        return loss, mse
