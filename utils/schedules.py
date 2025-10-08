import numpy as np


def linear_schedule(n, init=0.2, max_n=100000):
    """
    max_n: Reaches 1 only at the end of learning.
    """
    return min(1., init + (1 - init) * min(n / max_n, 1))


def epsilon_schedule_start_end(t: int, duration: int, init_value: float,
                               end_value: float, linear=True):
    if linear:
        slope = (end_value - init_value) / duration
        return max(slope * t + init_value, end_value)
    else:
        decay_rate = 3 / duration
        return max(end_value + (init_value - end_value) * (np.exp(-decay_rate * t)),
                   end_value)


def lr_schedule(optimizer, init_lr, global_step, total_timesteps,
                final_scale=0.1):
    r = (1 - final_scale)
    scale = 1 - (global_step / total_timesteps * r)
    new_lr = init_lr * scale
    optimizer.param_groups[0]["lr"] = new_lr
    return optimizer


