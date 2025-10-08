import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from models.networks import SimpleCNN


##################################################
# SAC with discrete action space
##################################################
class DiscreteSAC(nn.Module):
    def __init__(self, state_shape, action_shape, backbone_type="FC",
                 device="cpu", latent_dim=256):
        super().__init__()

        self.device = device
        self.backbone_type = backbone_type
        self.visual_feat_dim = None

        self.backbone = self._build_backbone(backbone_type)
        self._build_model(state_shape, action_shape, backbone_type, latent_dim,
                          device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

    def _build_backbone(self, backbone_type):
        if backbone_type == "CNN":
            self.visual_feat_dim = 64
            backbone = SimpleCNN(self.visual_feat_dim).to(self.device)

        else:
            self.visual_feat_dim = None
            backbone = nn.Sequential()

        return backbone

    def _build_model(self, state_shape, action_shape, backbone_type,
                     latent_dim, device):
        self.actor = DiscreteActor(state_shape, action_shape, backbone_type,
                                   self.visual_feat_dim,
                                   latent_dim).to(device)
        self.qf1 = DiscreteSoftQNetwork(
            state_shape, action_shape, backbone_type,
            self.visual_feat_dim, latent_dim).to(device)
        self.qf2 = DiscreteSoftQNetwork(
            state_shape, action_shape, backbone_type,
            self.visual_feat_dim, latent_dim).to(device)

        self.qf1_target = DiscreteSoftQNetwork(
            state_shape, action_shape, backbone_type,
            self.visual_feat_dim, latent_dim).to(device)
        self.qf2_target = DiscreteSoftQNetwork(
            state_shape, action_shape, backbone_type,
            self.visual_feat_dim, latent_dim).to(device)

    def get_action(self, x, with_exploration=True, return_probs=False):
        x = self.backbone(x)

        if with_exploration:
            action, log_prob, action_probs = self.actor.get_action(x)
            return action, log_prob, action_probs
        else:
            if return_probs:
                action, action_probs = self.actor.get_optimal_action(x, return_probs)
                return action, action_probs

            action = self.actor.get_optimal_action(x, return_probs)
            return action

    def get_q_values(self, x, a=None):
        x = self.backbone(x)

        qf1 = self.qf1.forward(x)
        qf2 = self.qf2.forward(x)
        return qf1, qf2

    def get_q_values_target_network(self, x, a=None):
        x = self.backbone(x)

        qf1_target = self.qf1_target.forward(x)
        qf2_target = self.qf2_target.forward(x)
        return qf1_target, qf2_target

    def load_pretrained_agent(self, path_to_checkpoint):
        checkpoint = torch.load(path_to_checkpoint, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        print(f"Load SAC agent from: {path_to_checkpoint}")


class DiscreteSoftQNetwork(nn.Module):
    def __init__(self, state_shape, action_shape, backbone_type="FC",
                 visual_feat_dim=None, latent_dim=64):
        super().__init__()

        state_shape = np.prod(state_shape)
        action_shape = np.prod(action_shape)
        self.backbone_type = backbone_type
        self.d_r = 0.1

        self.v_head = self._build_model(
            backbone_type, state_shape, visual_feat_dim, latent_dim,
            action_shape)

    def _build_model(self, backbone_type, state_shape, visual_feat_dim,
                     latent_dim, action_shape):
        if backbone_type == "CNN":
            v_head = nn.Sequential(
                nn.Linear(visual_feat_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, action_shape))
        else:
            v_head = nn.Sequential(
                nn.Linear(state_shape, latent_dim),
                nn.Dropout(self.d_r, inplace=True),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
                nn.Dropout(self.d_r, inplace=True),
                nn.ReLU(),
                nn.Linear(latent_dim, action_shape))
        return v_head

    def forward(self, x):
        v = self.v_head(x)
        return v


class DiscreteActor(nn.Module):
    def __init__(self, state_shape, action_shape, backbone_type="FC",
                 visual_feat_dim=None, latent_dim=64):
        super().__init__()

        action_shape = np.prod(action_shape)
        self.backbone_type = backbone_type
        self.d_r = 0.1

        state_shape = np.prod(state_shape)
        self.fc_logits = self._build_model(
            backbone_type, state_shape, visual_feat_dim, latent_dim,
            action_shape)

    def _build_model(self, backbone_type, state_shape, visual_feat_dim,
                     latent_dim, action_shape):
        if backbone_type == "CNN":
            fc_logits = nn.Sequential(
                nn.Linear(visual_feat_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, action_shape))

        else:
            fc_logits = nn.Sequential(
                nn.Linear(state_shape, latent_dim),
                nn.Dropout(self.d_r, inplace=True),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
                nn.Dropout(self.d_r, inplace=True),
                nn.ReLU(),
                nn.Linear(latent_dim, action_shape))
        return fc_logits

    def forward(self, x):
        logits = self.fc_logits(x)
        return logits

    def get_action(self, x):
        logits = self.forward(x)

        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        action_probs = policy_dist.probs

        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs

    def get_optimal_action(self, x, return_probs=False):
        logits = self.forward(x)
        if return_probs:
            action_probs = torch.softmax(logits, dim=-1)
            return torch.argmax(logits, dim=-1), action_probs

        return torch.argmax(logits, dim=-1)
