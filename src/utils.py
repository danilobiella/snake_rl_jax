import jax.numpy as jnp


def format_time(time_in_seconds: float) -> str:
    if time_in_seconds < 1e-6:
        return f"{time_in_seconds * 1e9:.1f} ns"
    elif time_in_seconds < 1e-3:
        return f"{time_in_seconds * 1e6:.1f} μs"
    elif time_in_seconds < 1:
        return f"{time_in_seconds * 1000:.1f} ms"
    elif time_in_seconds < 60:
        return f"{time_in_seconds:.2f} s"
    elif time_in_seconds < 3600:
        return f"{time_in_seconds / 60:.2f} min"
    elif time_in_seconds < 3600 * 24:
        return f"{time_in_seconds / 3600:.2f} h"
    else:
        return f"{time_in_seconds / 3600 / 24:.2f} d"


def repeat_tensor(tensor, N):
    return jnp.repeat(jnp.expand_dims(tensor, axis=0), repeats=N, axis=0)
