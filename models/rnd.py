import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks import SimpleCNN
from utils.utils import RunningMeanStd, initialize_weights_orthogonal


class RandomNetworkDistillation(nn.Module):
    def __init__(self, state_shape, feature_dim=256, backbone_type="FC",
                 device="cpu", norm_type="run_mean_std", ir_clip=2,
                 with_extra_obs=False):
        super().__init__()
        self.obs_hist = None
        self.device = device
        self.backbone_type = backbone_type

        self.with_extra_obs = with_extra_obs
        self.extra_obs_dim = 64
        self.extra_obs = torch.ones([1, self.extra_obs_dim],
                                    dtype=torch.float).to(self.device)

        self._build_model(state_shape, feature_dim)
        initialize_weights_orthogonal(self)

        self.norm_type = norm_type
        self.ir_clip = ir_clip
        if self.norm_type == "run_mean_std":
            self.ir_hist = RunningMeanStd(momentum=0.9)

        # Target network is not trainable
        for param in self.target_network.parameters():
            param.requires_grad = False

    def _build_model(self, state_shape, feature_dim):
        _dim = 64
        if self.backbone_type == "CNN":
            self.backbone = SimpleCNN(_dim).to(self.device)

            self.predictor = nn.Sequential(
                nn.Linear(_dim, _dim), nn.ReLU(),
                nn.Linear(_dim, _dim), nn.ReLU(),
                nn.Linear(_dim, feature_dim)).to(self.device)

            self.target_network = nn.Sequential(
                nn.Linear(_dim, _dim), nn.ReLU(),
                nn.Linear(_dim, _dim), nn.ReLU(),
                nn.Linear(_dim, feature_dim)).to(self.device)

        elif self.backbone_type == "FC":
            _dim = 16
            state_shape = np.prod(state_shape)
            if self.with_extra_obs:
                state_shape += self.extra_obs_dim

            self.backbone = nn.Sequential()
            self.predictor = nn.Sequential(
                nn.Linear(state_shape, _dim), nn.ReLU(),
                nn.Linear(_dim, _dim*2), nn.ReLU(),
                nn.Linear(_dim*2, feature_dim)).to(self.device)

            self.target_network = nn.Sequential(
                nn.Linear(state_shape, _dim), nn.ReLU(),
                nn.Linear(_dim, _dim*2), nn.ReLU(),
                nn.Linear(_dim*2, feature_dim)).to(self.device)

        else:
            raise NotImplementedError

    def forward(self, next_obs):
        if self.with_extra_obs:
            extra_obs = self.extra_obs.repeat(next_obs.shape[0], 1)
            next_obs = torch.cat([next_obs, extra_obs], dim=1)

        feat = self.backbone(next_obs)
        predict_feature = self.predictor(feat)
        with torch.no_grad():
            target_feature = self.target_network(feat)

        return predict_feature, target_feature

    def compute_novelty(self, x, terminations=None):
        predict_feature, target_feature = self.forward(x)
        err = F.mse_loss(predict_feature, target_feature, reduction='none'
                         ).mean(dim=1).cpu().detach().numpy()
        if terminations is not None:
            err = err * (1 - terminations)
        return err

    @torch.no_grad()
    def compute_intrinsic_reward(self, next_x, terminations=None, is_buffer=False):
        err = self.compute_novelty(next_x)

        if self.norm_type == "run_mean_std":
            if not is_buffer:
                self.ir_hist.update(err.reshape(-1))
            self.ir_mean = self.ir_hist.mean
            self.ir_std = self.ir_hist.std
            reward = (err - self.ir_mean) / (self.ir_std + 1e-8)

        elif self.norm_type == "batch_min_max":
            _min = np.min(err)
            _max = np.max(err)
            reward = (err - _min) / (_max - _min + 1e-8)
            self.ir_mean = np.mean(reward)
            self.ir_std = np.std(reward)

        else:
            self.ir_mean = err.mean()
            self.ir_std = err.std()
            reward = err

        reward[reward > self.ir_clip] = self.ir_clip
        reward[reward < -self.ir_clip] = -self.ir_clip

        if terminations is not None:
            # DeepSea, set terminated state IR to the maximum (easy to compute the weights)
            reward[terminations] = self.ir_clip

        info = {
            "intrinsic_reward_mean": self.ir_mean.copy(),
            "intrinsic_reward_std": self.ir_std.copy(),
            "intrinsic_reward": reward.copy()
        }
        return reward, info
