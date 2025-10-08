import torch

from models.kea import DiscreteKEA
from models.rnd import RandomNetworkDistillation
from trainers.sac_trainer import DiscreteSACTrainer


class DiscreteKEASACTrainer(DiscreteSACTrainer):
    def __init__(
        self,
        agent: DiscreteKEA,
        ir_agent: RandomNetworkDistillation,
        optimizers,
        log_alpha,
        target_entropy=None,
        device="cpu",
        result_dir="./",
        config={},
        log_alpha_co=None
    ):
        super().__init__(
            agent, ir_agent, optimizers, log_alpha,
            target_entropy, device, result_dir, config)

        self.log_alpha_co = (log_alpha_co if log_alpha_co is not None else
                             self.log_alpha.clone())     # Use for auto-tune
        self.alpha_co = log_alpha_co.exp().item()

        self.agent = agent

        if config.get("delay_standard_module_update", False):
            self.standard_module_loss_weight = 0
        else:
            self.standard_module_loss_weight = 1

    ##################################################
    # Update SAC
    ##################################################
    def update_rl_backbone(self, obs, actions, next_obs, ext_rewards,
                           int_rewards, dones, weights, exists=None):

        if self.ir_agent is not None:
            int_rewards = self._recompute_intrinsic_reward(
                next_obs, dones=None, obs=obs, exists=exists)
        else:
            int_rewards = torch.zeros_like(ext_rewards.flatten())

        # Update Q networks
        q_train_info = self.update_q_networks(
            obs, actions, next_obs, ext_rewards, int_rewards, dones, weights)

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
    def _compute_target_q(self, next_obs, rewards, dones, use_standard_modlue):
        alpha = self.alpha_co if use_standard_modlue else self.alpha

        next_state_actions, next_state_log_pi, next_state_action_probs = \
            self.agent.get_action_single_agent(next_obs, use_standard_modlue)
        qf1_next_target, qf2_next_target = \
            self.agent.get_q_values_target_network_single_agent(
                next_obs, next_state_actions, use_standard_modlue)
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)

        min_qf_next_target = \
            next_state_action_probs * (min_qf_next_target -
                                       alpha * next_state_log_pi)
        min_qf_next_target = min_qf_next_target.sum(dim=1).view(-1)
        next_q_value = (
            rewards + (1 - dones) * self.gamma * min_qf_next_target)

        return next_q_value

    def _compute_q_loss(self, next_q_value, obs, actions, weights, use_standard_modlue):
        # Compute Q_t
        qf1_values, qf2_values = self.agent.get_q_values_single_agent(
            obs, use_standard_modlue=use_standard_modlue)
        qf1_a_values = qf1_values.gather(1, actions.long()).view(-1)
        qf2_a_values = qf2_values.gather(1, actions.long()).view(-1)

        # Compute loss
        td_error_1 = next_q_value - qf1_a_values
        td_error_2 = next_q_value - qf2_a_values
        abs_td_error = torch.abs(td_error_1) + torch.abs(td_error_2)

        # For prioritized buffer replay (correct bias)
        qf1_loss = torch.mean(td_error_1.pow(2) * weights)
        qf2_loss = torch.mean(td_error_2.pow(2) * weights)
        qf_loss = qf1_loss + qf2_loss

        name = "standard_module" if use_standard_modlue else "novel_aug_module"
        info = {f"{name}_q1_values": qf1_a_values.mean().item(),
                f"{name}_q2_values": qf2_a_values.mean().item(),
                f"{name}_q1_loss":  qf1_loss.item(),
                f"{name}_q2_loss": qf2_loss.item()}
        return qf_loss, abs_td_error, info

    def update_q_networks(self, obs, actions, next_obs, ext_rewards,
                          int_rewards, dones, weights=None):

        # Compute SAC-int module loss
        rewards = self.config["extrinsic_reward_beta"] * ext_rewards + \
                  self.config["intrinsic_reward_beta"] * int_rewards
        next_q_value = self._compute_target_q(
            next_obs, rewards, dones, use_standard_modlue=False)
        qf_loss, abs_td_error, info = self._compute_q_loss(
            next_q_value, obs, actions, weights, use_standard_modlue=False)

        # Compute standard module loss
        if self.standard_module_loss_weight > 0:
            rewards = self.config["extrinsic_reward_beta"] * ext_rewards
            co_next_q_value = self._compute_target_q(
                next_obs, rewards, dones, use_standard_modlue=True)
            co_qf_loss, co_abs_td_error, co_info = self._compute_q_loss(
                co_next_q_value, obs, actions, weights, use_standard_modlue=True)

        else:
            co_qf_loss = torch.zeros_like(qf_loss)
            co_abs_td_error = 0
            co_info = {}

        # Update the model
        qf_loss = co_qf_loss + qf_loss

        self._gradient_descent(self.q_optimizer, qf_loss, is_clip_grad=False)

        abs_td_error = co_abs_td_error + abs_td_error

        info = {**info,
                **co_info,
                "q_loss": qf_loss.item(),
                "abs_td_error": abs_td_error.cpu().detach()}
        return info

    def update_target_newtorks(self, tau):
        if self.standard_module_loss_weight > 0:
            for param, target_param in zip(self.agent.standard_module.qf1.parameters(),
                                        self.agent.standard_module.qf1_target.parameters()):
                target_param.data.copy_(tau * param.data +
                                        (1 - tau) * target_param.data)

            for param, target_param in zip(self.agent.standard_module.qf2.parameters(),
                                        self.agent.standard_module.qf2_target.parameters()):
                target_param.data.copy_(tau * param.data +
                                        (1 - tau) * target_param.data)

        for param, target_param in zip(self.agent.novel_aug_module.qf1.parameters(),
                                       self.agent.novel_aug_module.qf1_target.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1 - tau) * target_param.data)

        for param, target_param in zip(self.agent.novel_aug_module.qf2.parameters(),
                                       self.agent.novel_aug_module.qf2_target.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1 - tau) * target_param.data)

    ##############################
    # Update SAC: Actor
    ##############################
    def _autotune_alpha(self, obs, weights, use_standard_modlue=False):
        with torch.no_grad():
            _, log_pi, action_probs = self.agent.get_action_single_agent(
                obs, use_standard_modlue=use_standard_modlue)

        if use_standard_modlue:
            alpha = self.log_alpha_co.exp()
        else:
            alpha = self.log_alpha.exp()

        loss = (action_probs.detach() *
                (-alpha * (log_pi + self.target_entropy).detach())
               ).mean(dim=1)
        loss = (loss * weights).mean()

        self._gradient_descent(self.a_optimizer, loss, is_clip_grad=False)

        if use_standard_modlue:
            self.alpha_co = self.log_alpha_co.exp().item()
        else:
            self.alpha = self.log_alpha.exp().item()

        loss = loss.item()
        return loss

    def _compute_actor_loss(self, obs, alpha, weights, use_standard_modlue):
        actions, log_pi, action_probs = self.agent.get_action_single_agent(
            obs, use_standard_modlue)

        qf1_values, qf2_values = self.agent.get_q_values_single_agent(
            obs, use_standard_modlue=use_standard_modlue)
        min_qf_values = torch.min(qf1_values, qf2_values)

        # Compute loss
        actor_loss = (action_probs *
                      ((alpha * log_pi) - min_qf_values)).mean(dim=1)
        actor_loss = (actor_loss * weights).mean()
        return actor_loss

    def update_actor_network(self, obs, actions, weights=None):
        for _ in range(self.config["train_policy_freq"]):

            # Compute SAC-int module loss
            novel_aug_module_actor_loss = self._compute_actor_loss(
                obs, self.alpha, weights, use_standard_modlue=False)

            # Compute standard module loss
            if self.standard_module_loss_weight > 0:
                co_actor_loss = self._compute_actor_loss(
                    obs, self.alpha_co, weights, use_standard_modlue=True)
            else:
                co_actor_loss = torch.zeros_like(novel_aug_module_actor_loss)

            actor_loss = self.standard_module_loss_weight * co_actor_loss + novel_aug_module_actor_loss

            # Update the model
            self._gradient_descent(self.actor_optimizer, actor_loss,
                                   is_clip_grad=False)

            # Autotune alpha
            alpha_loss, alpha_loss_co = 0, 0
            if self.config.get("autotune", False):
                alpha_loss = self._autotune_alpha(obs, weights, use_standard_modlue=False)

        info = {"loss": actor_loss.item(),
                "novel_aug_module_actor_loss": novel_aug_module_actor_loss.item(),
                "standard_module_actor_loss": co_actor_loss.item(),
                "alpha_loss": alpha_loss,
                "alpha_loss_co": alpha_loss_co,
                "alpha": self.alpha,
                "alpha_co": self.alpha_co}
        return info
