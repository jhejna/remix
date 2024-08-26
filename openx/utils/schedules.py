import optax
from jax import numpy as jnp


def warmup_rsqrt_schedule(init_value: float, peak_value: float, warmup_steps: int, timescale: int = 10000):
    return optax.join_schedules(
        [
            optax.linear_schedule(
                init_value=init_value,
                end_value=peak_value,
                transition_steps=warmup_steps,
            ),
            lambda step: peak_value / jnp.sqrt((step + timescale) / timescale),
        ],
        [warmup_steps],
    )
