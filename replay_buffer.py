from abc import ABC, abstractmethod
from gym import spaces              # Used for DeepSea
import numpy as np
import pickle
import torch
from typing import Any, List, Dict, Tuple, Union, NamedTuple
import warnings

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


""" Reference from stable-baselines3
    (https://github.com/DLR-RM/stable-baselines3)
    NOTE: Stable-baselines3 doesn't support new gymnasium version
"""


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def get_obs_shape(
    observation_space: spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    intrinsic_rewards: torch.Tensor


class PrioritizedReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    # img_observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    # next_img_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    intrinsic_rewards: torch.Tensor
    exists: torch.Tensor
    batch_inds: np.ndarray
    env_inds: np.ndarray
    weights: torch.Tensor


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: str = "cuda",
        n_envs: int = 1
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        # self.device = torch.device(device)
        self.device = device
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def real_size(self) -> int:
        """ Return the real current buffer size. """
        if self.full:
            return (self.buffer_size * self.n_envs)
        return int(self.pos * self.n_envs)

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env=None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray, env=None
    ) -> Union[ReplayBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        env=None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env=None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward


class PrioritizedReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3 with priorization.
    Ref: https://github.com/vwxyzjn/cleanrl &
         https://github.com/DLR-RM/stable-baselines3

    buffer_size: Max number of element in the buffer
    observation_space: Observation space
    action_space: Action space
    device: PyTorch device
    n_envs: Number of parallel environments
    alpha: How much priorization is used (0: disabled, 1: full priorization)
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: str = "cuda",
        n_envs: int = 1,
        alpha: float = 1,
        with_dict_obs: bool = False
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.with_dict_obs = with_dict_obs
        self.alpha = alpha

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        if not with_dict_obs:
            self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.intrinsic_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.exists = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Priorities
        self.priorities = np.ones((self.buffer_size, self.n_envs), dtype=np.float32)
        self._min_priority = 1e-5

        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # self._check_enough_memory(psutil, mem_available)

    def _check_enough_memory(self, psutil, mem_available):
        if psutil is not None:
            total_memory_usage = \
                self.observations.nbytes + self.actions.nbytes + \
                self.rewards.nbytes + self.intrinsic_rewards.nbytes + \
                self.dones.nbytes + self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        intrinsic_rewards: np.ndarray,
        done: np.ndarray,
        exists: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        if not self.with_dict_obs:
            self.observations[self.pos] = np.array(obs).copy()
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.intrinsic_rewards[self.pos] = np.array(intrinsic_rewards).copy()
        self.dones[self.pos] = np.array(done).copy()

        if exists is not None:
            self.exists[self.pos] = np.array(exists).copy()

        if self.full:
            priority = np.max(self.priorities, axis=0)
        elif self.real_size() != 0:
            priority = np.max(self.priorities[:self.pos], axis=0)
        else:
            priority = np.ones([self.n_envs], dtype=np.float32)

        self.priorities[self.pos] = priority

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _get_samples(self, batch_inds: np.ndarray, env_inds: np.ndarray, env=None) -> Tuple:
        if not self.with_dict_obs:
            data = (
                self._normalize_obs(self.observations[batch_inds, env_inds, :], env),
                self.actions[batch_inds, env_inds, :],
                self._normalize_obs(self.next_observations[batch_inds, env_inds, :], env),
                self.dones[batch_inds, env_inds].reshape(-1, 1),
                self._normalize_reward(self.rewards[batch_inds, env_inds].reshape(-1, 1), env),
                self.intrinsic_rewards[batch_inds, env_inds].reshape(-1, 1),
                self.exists[batch_inds, env_inds].reshape(-1, 1),
            )

        return data

    def last_samples(self, batch_size):
        bs = batch_size // self.n_envs

        if not self.with_dict_obs:
            s_idx, e_idx = self.pos-bs, self.pos
            if s_idx < 0:
                return (np.concatenate([self.next_observations[s_idx:, ...],
                                        self.next_observations[:e_idx, ...]], axis=0),
                        np.concatenate([self.observations[s_idx:, ...],
                                        self.observations[:e_idx, ...]], axis=0),
                        np.concatenate([self.actions[s_idx:, ...],
                                        self.actions[:e_idx, ...]], axis=0),
                        np.concatenate([self.dones[s_idx:, ...],
                                        self.dones[:e_idx, ...]], axis=0),
                        np.concatenate([self.exists[s_idx:, ...],
                                        self.exists[:e_idx, ...]], axis=0))

            return (self.next_observations[s_idx:e_idx, ...],
                    self.observations[s_idx:e_idx, ...],
                    self.actions[s_idx:e_idx, ...],
                    self.dones[s_idx:e_idx, ...],
                    self.exists[s_idx:e_idx, ...])

    def sample(self, batch_size: int, beta: float = 0, env=None, random=False
               ) -> PrioritizedReplayBufferSamples:
        """
        Sample elements from the replay buffer using priorization.

        batch_size  : Number of element to sample
        env         : associated gym VecEnv to normalize the observations/rewards when sampling
        """
        if self.alpha == 0 or random:
            upper_bound = self.buffer_size if self.full else self.pos
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)
            env_inds = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
            data = self._get_samples(batch_inds, env_inds, env=env)

            weights = torch.ones([batch_size], dtype=torch.float,
                                 device=self.device)
            experiences = PrioritizedReplayBufferSamples(
                *tuple(map(self.to_torch, data)), batch_inds=batch_inds,
                env_inds=env_inds, weights=weights)
            return experiences

        # Sampling probabilty
        priorities = self.priorities[:self.pos, :] \
                   if not self.full else self.priorities
        sample_probs = priorities**self.alpha / np.sum(priorities**self.alpha)

        # Sampling
        flat_indices = np.random.choice(
            np.arange(sample_probs.reshape(-1).shape[0]), size=batch_size,
            p=sample_probs.reshape(-1), replace=True)
        batch_inds, env_inds = np.unravel_index(flat_indices, priorities.shape)
        data = self._get_samples(batch_inds, env_inds, env=env)

        # Compute correction weights
        N = self.real_size()
        weights = (N * sample_probs[batch_inds, env_inds]) ** (-beta)
        weights = weights / weights.max()

        if not self.with_dict_obs:
            experiences = PrioritizedReplayBufferSamples(
                *tuple(map(self.to_torch, data)), batch_inds=batch_inds,
                env_inds=env_inds, weights=torch.tensor(weights, device=self.device))

        return experiences

    def update_priorities(
        self,
        batch_inds: np.ndarray,
        env_inds: np.ndarray,
        priorities: np.ndarray
    ) -> None:

        if not np.isfinite(priorities).all() or (priorities < 0.0).any():
            raise ValueError("[PrioritizedReplayBuffer] Priorities must be finite and positive.")

        self.priorities[batch_inds, env_inds] = priorities + self._min_priority

    def save(self, path: str) -> None:
        data = {
            "buffer_size": self.buffer_size,
            "n_envs": self.n_envs,
            "pos": self.pos,
            "observations": self.observations,
            "next_observations": self.next_observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "intrinsic_rewards": self.intrinsic_rewards,
            "dones": self.dones,
            "exists": self.exists,
            "priorities": self.priorities
        }
        with open(path, 'wb') as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
        print(f"Saved buffer in {path}")

    def load(self, path: str = "", data: Dict = {}) -> None:
        if len(data) == 0:
            print(f"Load buffer from {path}")
            with open(path, 'rb') as file:
                data = pickle.load(file)
        if data["pos"] > 0:
            self.pos = data["pos"]
            self.observations[:self.pos] = data["observations"][:self.pos]
            self.next_observations[:self.pos] = data["next_observations"][:self.pos]
            self.actions[:self.pos] = data["actions"][:self.pos]
            self.intrinsic_rewards[:self.pos] = data["intrinsic_rewards"][:self.pos]
            self.rewards[:self.pos] = data["rewards"][:self.pos]
            self.dones[:self.pos] = data["dones"][:self.pos]
            self.exists[:self.pos] = data["exists"][:self.pos]
            self.priorities[:self.pos] = data["priorities"][:self.pos]
