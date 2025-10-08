import torch
import torch.nn as nn
from models.sac import DiscreteSAC


##################################################
# KEA with discrete action space
##################################################
class DiscreteKEA(nn.Module):
    def __init__(self, state_shape, action_shape, backbone_type="FC",
                 device="cpu", latent_dim=256, rl_agent_name="SAC"):
        super().__init__()

        # standard agent AS and the novelty-augmented agent
        self.device = device
        if "SAC" in rl_agent_name:
            self.standard_module = DiscreteSAC(
                state_shape, action_shape, backbone_type, device, latent_dim)
            self.novel_aug_module = DiscreteSAC(
                state_shape, action_shape, backbone_type, device, latent_dim)

        else:
            raise NotImplementedError(f"{rl_agent_name} is not supported!")

    def get_action_single_agent(self, x, use_standard_modlue):
        sac = self.standard_module if use_standard_modlue else self.novel_aug_module
        action, log_prob, action_probs = sac.get_action(x)
        return action, log_prob, action_probs

    def get_q_values_single_agent(self, x, a=None, use_standard_modlue=False):
        sac = self.standard_module if use_standard_modlue else self.novel_aug_module
        return sac.get_q_values(x, a)

    def get_q_values_target_network_single_agent(self, x, a=None, use_standard_modlue=False):
        sac = self.standard_module if use_standard_modlue else self.novel_aug_module
        return sac.get_q_values_target_network(x, a)

    def get_action(self, x, with_exploration=True, use_standard_modlue=None, return_probs=False):

        if with_exploration:
            co_action, co_log_prob, co_action_prob = \
                self.standard_module.get_action(x, with_exploration)

            action, log_prob, action_prob = \
                self.novel_aug_module.get_action(x, with_exploration)

            # NOTE: In descrete, the actions shape is (bs)
            # novel_aug_module if novel == 1; standard_module if novel == 0
            action = (1 - use_standard_modlue.squeeze()) * action + use_standard_modlue.squeeze() * co_action
            log_prob = (1 - use_standard_modlue) * log_prob + use_standard_modlue * co_log_prob
            action_prob = (1 - use_standard_modlue) * action_prob + use_standard_modlue * co_action_prob

            return action, log_prob, action_prob
        else:
            if use_standard_modlue == 0:
                if return_probs:
                    action, action_probs = self.standard_module.get_action(x, with_exploration, return_probs)
                    return action, action_probs
                action = self.standard_module.get_action(x, with_exploration, return_probs)

            elif use_standard_modlue == 1:
                if return_probs:
                    action, action_probs = self.novel_aug_module.get_action(x, with_exploration, return_probs)
                    return action, action_probs
                action = self.novel_aug_module.get_action(x, with_exploration, return_probs)
            return action

    def load_pretrained_agent(self, path_to_checkpoint):
        checkpoint = torch.load(path_to_checkpoint, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        print(f"Load SAC agent from: {path_to_checkpoint}")
