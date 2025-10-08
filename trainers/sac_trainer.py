import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sac import DiscreteSAC
from models.rnd import RandomNetworkDistillation


class BaseTrainer(nn.Module):
    def __init__(
        self,
        agent,
        log_alpha,
        optimizers,
        target_entropy=None,
        device="cpu",
        result_dir="./",
        config={}
    ):
        super().__init__()
        self.agent = agent
        self.log_alpha = log_alpha
        self.alpha = log_alpha.exp().item() if log_alpha is not None else 0
        self.q_optimizer = optimizers["q_optimizer"]
        self.actor_optimizer = optimizers["actor_optimizer"]
        self.a_optimizer = optimizers["a_optimizer"]
        self.device = device
        self.result_dir = result_dir
        self.target_entropy = target_entropy
        self.train_steps = 0

        self.config = config
        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.train_policy_freq = config["train_policy_freq"]
        self.target_network_update_freq = config["target_network_update_freq"]

    def _clip_grad(self, agent_type):
        if agent_type == "policy":
            torch.nn.utils.clip_grad_norm_(
                self.agent.parameters(), self.max_grad_norm)
        elif agent_type == "ir_agent":
            torch.nn.utils.clip_grad_norm_(
                self.ir_agent.parameters(), self.max_grad_norm)

    def _gradient_descent(self, optimizer, loss, agent_type=None,
                          is_clip_grad=None):
        optimizer.zero_grad()
        loss.backward()

        if is_clip_grad:
            self._clip_grad(agent_type)
        optimizer.step()

    ##################################################
    # Update SAC: Q-Networks
    ##################################################
    def update_q_networks(self, data):
        raise NotImplementedError

    def update_target_newtorks(self, tau):
        raise NotImplementedError

    ##################################################
    # Update SAC: Actor
    ##################################################
    def update_actor_network(self, data):
        raise NotImplementedError


class DiscreteSACTrainer(BaseTrainer):
    def __init__(
        self,
        agent: DiscreteSAC,
        ir_agent: RandomNetworkDistillation,
        optimizers,
        log_alpha,
        target_entropy=None,
        device="cpu",
        result_dir="./",
        config={}
    ):
        super().__init__(
            agent, log_alpha, optimizers,
            target_entropy, device, result_dir, config)

        self.ir_agent = ir_agent
        self.ir_optimizer = optimizers["ir_optimizer"]

        self.max_grad_norm = 0.5
        if config["backbone_type"] == "FC":
            self._is_obs_image = False
        elif config["backbone_type"] == "CNN":
            self._is_obs_image = True

    def _obs_preprocess(self, obs: torch.tensor, is_image=True):
        """ img_obs: A torch tensor with (bs, g, w, channel) in unit8.
            Process it into a torch tensor with (bs, channel, h, w) in float32 ([0, 1]).
        """
        if self._is_obs_image:
            return obs.type(torch.float).to(self.device).permute(0, 3, 1, 2)
        else:
            return obs.type(torch.float).to(self.device)

    def _preprocess(self, data):
        if self._is_obs_image:
            obs = data.observations.to(self.device).permute(0, 3, 1, 2)
            next_obs = data.next_observations.to(self.device).permute(0, 3, 1, 2)
        else:
            obs = data.observations.to(self.device)
            next_obs = data.next_observations.to(self.device)

        actions = data.actions.to(self.device)
        ext_rewards = data.rewards.flatten()
        int_rewards = data.intrinsic_rewards.flatten()
        dones = data.dones.flatten()
        weights = data.weights.view(-1)

        exists = data.exists
        if exists is not None:
            exists = exists.flatten()

        return obs, actions, next_obs, ext_rewards, int_rewards, dones, weights, exists

    def train_once(self, data):
        """ Main entrence of the training loop. """

        self.train_steps += 1

        # Preprocess data
        (obs, actions, next_obs, ext_rewards, int_rewards, dones,
         weights, exists) = self._preprocess(data)

        # Update SAC
        info_rl_backbone = self.update_rl_backbone(
            obs, actions, next_obs, ext_rewards, int_rewards, dones,
            weights, exists=exists)

        return {**info_rl_backbone}

    ##################################################
    # Update intrinsic reward agent
    ##################################################
    def _compute_forward_loss(self, next_s, terminations=None):
        (predict_next_state_feature,
         target_next_state_feature) = self.ir_agent.forward(next_s)

        forward_loss = F.mse_loss(
            predict_next_state_feature,
            target_next_state_feature.detach(), reduction="none").mean(dim=1)

        if terminations is None:
            return forward_loss.mean()

        # Set zeros for termination states
        forward_loss = forward_loss * (1 - terminations)
        n_samples = (1 - terminations).sum()
        forward_loss = forward_loss.sum() * (1 / n_samples if n_samples > 0 else 0.0)
        return forward_loss

    def on_policy_update_ir_agent(self, s, next_s=None, terminations=None):
        loss = self._compute_forward_loss(s, terminations)
        if next_s is not None:
            loss = 0.5 * (loss + self._compute_forward_loss(next_s, terminations))

        self._gradient_descent(self.ir_optimizer, loss, "ir_agent",
                               is_clip_grad=True)

        info = {"rnd_loss": loss.detach().mean().cpu().item()}
        return info

    def update_intrinsic_reward_model(self, data):
        loss = self._compute_forward_loss(
            self._obs_preprocess(data.observations, is_image=True))
        self._gradient_descent(self.ir_optimizer, loss, "ir_agent",
                               is_clip_grad=False)

        info = {"rnd_loss": loss.detach().mean().cpu().item()}
        return info

    ##################################################
    # Update SAC
    ##################################################
    def _recompute_intrinsic_reward(self, next_obs, dones, obs=None, exists=None):
        if self.config.get("IR_model") == "RND":
            int_rewards, int_r_info = self.ir_agent.compute_intrinsic_reward(
                next_obs, terminations=dones, is_buffer=True)

        int_rewards = torch.from_numpy(int_rewards).to(self.device)

        return int_rewards

    def update_rl_backbone(self, obs, actions, next_obs, ext_rewards,
                           int_rewards, dones, weights, exists=None):
        if self.ir_agent is not None:
            int_rewards = self._recompute_intrinsic_reward(next_obs, dones=None, obs=obs, exists=exists)
        else:
            int_rewards = torch.zeros_like(ext_rewards.flatten())

        rewards = self.config["extrinsic_reward_beta"] * ext_rewards + \
                  self.config["intrinsic_reward_beta"] * int_rewards

        # Update Q networks
        q_train_info = self.update_q_networks(
            obs, actions, next_obs, rewards, dones, weights)

        # Update actor network
        actor_train_info = {}
        if self.train_steps % self.train_policy_freq == 0:
            actor_train_info = self.update_actor_network(obs, actions, weights)

        # Update the target networks
        if self.train_steps % self.target_network_update_freq == 0:
            self.update_target_newtorks(self.tau)

        return {"q_network": q_train_info,
                "actor": actor_train_info,
                "recomputed_int_reward": int_rewards.mean().item()}

    ##############################
    # Update SAC: Q-Networks
    ##############################
    @torch.no_grad()
    def _compute_target_q(self, next_obs, rewards, dones):
        next_state_actions, next_state_log_pi, next_state_action_probs = \
            self.agent.get_action(next_obs)
        qf1_next_target, qf2_next_target = \
            self.agent.get_q_values_target_network(
                next_obs, next_state_actions)
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)

        min_qf_next_target = \
            next_state_action_probs * (min_qf_next_target -
                                       self.alpha * next_state_log_pi)
        min_qf_next_target = min_qf_next_target.sum(dim=1).view(-1)
        next_q_value = (
            rewards + (1 - dones) * self.gamma * min_qf_next_target)

        return next_q_value

    def update_q_networks(self, obs, actions, next_obs, rewards, dones, weights=None):

        # compute target Q
        next_q_value = self._compute_target_q(next_obs, rewards, dones)

        # Compute Q_t
        qf1_values, qf2_values = self.agent.get_q_values(obs)
        qf1_a_values = qf1_values.gather(1, actions.long()).view(-1)
        qf2_a_values = qf2_values.gather(1, actions.long()).view(-1)

        # Compute loss
        td_error_1 = next_q_value - qf1_a_values
        td_error_2 = next_q_value - qf2_a_values
        abs_td_error = (torch.abs(td_error_1) + torch.abs(td_error_2)) / 2.

        # For prioritized buffer replay (correct bias)
        qf1_loss = torch.mean(td_error_1.pow(2) * weights)
        qf2_loss = torch.mean(td_error_2.pow(2) * weights)
        qf_loss = qf1_loss + qf2_loss

        # Update the model
        self._gradient_descent(self.q_optimizer, qf_loss, is_clip_grad=False)

        info = {"q1_values": qf1_a_values.mean().item(),
                "q2_values": qf2_a_values.mean().item(),
                "q1_loss":  qf1_loss.item(),
                "q2_loss": qf2_loss.item(),
                "q_loss": qf_loss.item() / 2.,
                "abs_td_error": abs_td_error.cpu().detach()}
        return info

    def update_target_newtorks(self, tau):
        for param, target_param in zip(self.agent.qf1.parameters(),
                                       self.agent.qf1_target.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1 - tau) * target_param.data)

        for param, target_param in zip(self.agent.qf2.parameters(),
                                       self.agent.qf2_target.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1 - tau) * target_param.data)

    ##############################
    # Update SAC: Actor
    ##############################
    def _autotune_alpha(self, obs, weights):
        with torch.no_grad():
            _, log_pi, action_probs = self.agent.get_action(obs)

        alpha = self.log_alpha.exp()
        loss = (action_probs.detach() *
                (-alpha * (log_pi + self.target_entropy).detach())
               ).mean(dim=1)
        loss = (loss * weights).mean()

        self._gradient_descent(self.a_optimizer, loss, is_clip_grad=False)

        self.alpha = self.log_alpha.exp().item()
        loss = loss.item()
        return loss

    def update_actor_network(self, obs, actions, weights=None):
        for _ in range(self.config["train_policy_freq"]):
            actions, log_pi, action_probs = self.agent.get_action(obs)

            qf1_values, qf2_values = self.agent.get_q_values(obs, actions)
            min_qf_values = torch.min(qf1_values, qf2_values)

            # Compute loss
            actor_loss = (action_probs *
                          ((self.alpha * log_pi) - min_qf_values)).mean(dim=1)
            actor_loss = (actor_loss * weights).mean()

            # Update the model
            self._gradient_descent(self.actor_optimizer, actor_loss,
                                   is_clip_grad=False)

            # Autotune alpha
            alpha_loss = 0
            if self.config.get("autotune", False):
                alpha_loss = self._autotune_alpha(obs, weights)

        info = {"loss": actor_loss.item(),
                "alpha_loss": alpha_loss,
                "alpha": self.alpha,
                "alpha_entropy": (-self.alpha * log_pi).mean().item()}
        return info
