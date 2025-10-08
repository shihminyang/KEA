# Used for DeepSea
import bsuite
from bsuite.utils import gym_wrapper
import gym

import mujoco
import numpy as np
import random
import torch


def make_env(env_id, seed, kwargs, wrapper_list=[], capture_video=False, run_name="video"):
    # [code release] Used
    def thunk():
        if "deep_sea" in env_id:
            env = bsuite.load_from_id(env_id)
            env = gym_wrapper.GymFromDMEnv(env)
            if seed is not None:
                print("Use seed:", seed)
                env.seed(seed)
        else:
            env = gym.make(env_id, **kwargs)

        if capture_video:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

        for wrapper in wrapper_list:
            env = wrapper(env)

        # env.action_space.seed(seed)
        return env

    return thunk


def set_random_seed(seed):
    # [code release] Used
    seed = seed if seed >= 0 else random.randint(0, 2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
