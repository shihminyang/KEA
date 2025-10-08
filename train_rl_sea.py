import argparse
import gym
import numpy as np
import os
from os import path
import shimmy           # Will register environments
import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.sac import DiscreteSAC
from models.kea import DiscreteKEA
from models.rnd import RandomNetworkDistillation
from replay_buffer import PrioritizedReplayBuffer
from trainers.kea_trainer import DiscreteKEASACTrainer
from utils.schedules import lr_schedule, linear_schedule
from utils.utils import load_and_parse_configs
from utils.sim import set_random_seed, make_env
from utils.reporter import Reporter


def decay_lr(optimizers, global_step, total_timesteps, config):
    if optimizers["q_optimizer"] is not None:
        lr_schedule(optimizers["q_optimizer"],
                    config["critic_learning_rate"], global_step,
                    total_timesteps, config["decay_lr_final_scale"])

    if optimizers["actor_optimizer"] is not None:
        lr_schedule(optimizers["actor_optimizer"],
                    config["actor_learning_rate"], global_step,
                    total_timesteps, config["decay_lr_final_scale"])

    if optimizers["a_optimizer"] is not None:
        lr_schedule(optimizers["a_optimizer"],
                    config["critic_learning_rate"], global_step,
                    total_timesteps, config["decay_lr_final_scale"])

    if optimizers["ir_optimizer"] is not None:
        lr_schedule(optimizers["ir_optimizer"],
                    config["ir_learning_rate"], global_step,
                    total_timesteps, config["decay_lr_final_scale"])


class Runner:
    def __init__(
        self,
        envs,
        agent: DiscreteSAC | DiscreteKEA,
        ir_agent: RandomNetworkDistillation,
        trainer: DiscreteKEASACTrainer,
        rb: PrioritizedReplayBuffer,
        reporter: Reporter,
        config,
        result_dir: str,
        device: str,
        num_envs: int,
        num_eval_envs: int = 2,
        val_envs=None,
        seeds=None,
        render_mode: str = "rgb_array"
    ):
        self.seeds = seeds
        self.env_id = config.get("env_id")
        self.reward_threshold = config.get("reward_threshold", 0.25)
        self.envs = envs
        self.val_envs = val_envs
        self.agent = agent
        self.ir_agent = ir_agent
        self.trainer = trainer
        self.rb = rb
        self.reporter = reporter
        self.device = device
        self.n_transitions = 0

        self.gamma = config["gamma"]
        self.num_envs = num_envs
        self.num_eval_envs = num_eval_envs
        self.render_mode = render_mode

        self.result_dir = result_dir

        self.obs_type = config["observation_type"]
        self.with_int_rewards = config["with_intrinsic_rewards"]

        self.eipsode_rewards_tracking = np.zeros(num_envs)      # For compute a sparse reward
        self.eipsode_rewards = np.zeros(num_envs)
        self.eipsode_returns = np.zeros(num_envs)
        self.eipsode_lengths = np.zeros(num_envs, dtype=int)
        self.eipsode_int_rewards = np.zeros(num_envs)
        self.eipsode_lengths_eval = np.zeros(num_eval_envs, dtype=int)

        self.buffer_init_beta = config["buffer_init_beta"]
        self.total_timesteps = config["total_timesteps"]//num_envs
        self.training_intensity = config["training_intensity"]
        self.train_intrinsic_intensity = config["train_intrinsic_intensity"]
        self.num_eval_episodes = config["num_eval_episodes"]
        self.batch_size = config["batch_size"]

        self.num_samples_before_learning = config["num_samples_before_learning"]//num_envs
        self.train_freq = max(config["train_freq"]//num_envs, 1)
        self.eval_freq = max(config["eval_freq"]//num_envs, 1)
        self.train_ir_freq = max(config["train_ir_freq"]//num_envs, 1)
        self.save_model_freq = config["save_model_freq"]//num_envs
        self.report_rollout_freq = config["report_rollout_freq"]//num_envs
        self.report_loss_freq = config["report_loss_freq"]//num_envs

        self.num_samples_before_evaluation = config["num_samples_before_evaluation"]//num_envs

        self.horizon = config["max_episode_steps"]

        self.hist_images = []

        self.config = config
        self.use_standard_module = None
        self.use_standard_module_eval = None
        if "KEA" in config["rl_agent"]:
            self.use_standard_module = torch.zeros(num_envs, 1).to(device)
            self.use_standard_module_eval = torch.ones(num_eval_envs, 1).to(device)

    ##################################################
    # Info
    ##################################################
    def _update_episode_info(self, rewards, int_rewards):
        self.eipsode_rewards += rewards
        self.eipsode_returns += (rewards * self.gamma**(self.eipsode_lengths))
        self.eipsode_lengths += 1
        self.eipsode_int_rewards += int_rewards

    def _record_episode_info(self, dones, infos):
        reward_list = self.eipsode_rewards[dones]
        return_list = self.eipsode_returns[dones]
        length_list = self.eipsode_lengths[dones]
        int_reward_list = self.eipsode_int_rewards[dones]

        self.reporter["total_return"] += return_list.tolist()
        self.reporter["total_reward"] += reward_list.tolist()
        self.reporter["episode_length"] += length_list.tolist()
        self.reporter["total_int_reward"] += int_reward_list.tolist()
        if "is_success" in infos:
            self.reporter["hist_complete"] += infos['is_success'][dones].tolist()
        else:
            self.reporter["hist_complete"] += (reward_list > 0).tolist()

    def _reset_episode_info(self, dones):
        self.eipsode_rewards[dones] = 0
        self.eipsode_returns[dones] = 0
        self.eipsode_lengths[dones] = 0
        self.eipsode_int_rewards[dones] = 0

    ##################################################
    # Rollout
    ##################################################
    def _preprocess(self, obs):
        if self.obs_type == "image":
            return torch.FloatTensor(obs).to(self.device).permute(0, 3, 1, 2)
        else:
            return torch.FloatTensor(obs).to(self.device).type(torch.float)

    def _random_actions(self):
        actions = np.array([self.envs.single_action_space.sample()
                            for _ in range(self.envs.num_envs)])
        return actions

    def _estimate_actions(self, obs):
        if self.use_standard_module is None:
            actions, log_probs, action_probs = self.agent.get_action(
                self._preprocess(obs))
        else:
            actions, log_probs, action_probs = self.agent.get_action(
                self._preprocess(obs), use_standard_module=self.use_standard_module)

        actions = actions.detach().cpu().numpy()

        entropies = (-action_probs * torch.log(action_probs+1e-11)
                    ).sum(axis=1).detach().cpu().numpy()
        return actions, entropies, log_probs

    def compute_inteinsic_rewards(self, next_obs, obs=None, terminations=None):
        if self.config["IR_model"] == "RND":
            int_rewards, int_r_info = self.ir_agent.compute_intrinsic_reward(
                self._preprocess(next_obs), terminations=terminations)

        return int_rewards, int_r_info

    def _switching_mechanism(self, int_rewards):
        # Enter a novel state, using standard_module for random search
        use_standard_module_mask = int_rewards > self.config["switch_threshold"]
        self.use_standard_module[use_standard_module_mask] = 1.
        self.use_standard_module[np.logical_not(use_standard_module_mask)] = 0.

    def step(self, obs, global_step):
        if self.is_learning_starts(global_step):
            actions, entropies, log_probs = self._estimate_actions(obs)
            self.reporter["entropy"] += entropies.tolist()
        else:
            actions = self._random_actions()

        next_obs, rewards, dones, infos = self.envs.step(actions)

        if self.with_int_rewards:
            int_rewards, int_r_info = self.compute_inteinsic_rewards(next_obs, obs)
        else:
            int_r_info = {}
            int_rewards = np.zeros_like(rewards)

        # Update novels
        if "KEA" in self.config["rl_agent"]:
            self._switching_mechanism(int_rewards)
            self.reporter["num_use_novel_aug_module"] += \
                (1 - self.use_standard_module.detach().cpu().numpy()).tolist()

        # Add to buffer
        self.rb.add(obs, next_obs, actions, rewards, int_rewards,
                    dones, int_r_info.get("exists"), infos)

        self.n_transitions += obs.shape[0]

        # Update episode info
        self._update_episode_info(rewards, int_rewards)

        if dones.any():
            # In Gym (DeepSea), the autoreset mode is in Disabled mode so the next_obs get from reset
            self._record_episode_info(dones, infos)
            self.eipsode_rewards_tracking[dones] = 0

            self._reset_episode_info(dones)
            if self.config.get("delay_standard_module_update", False) and \
               (self.eipsode_rewards > 0).any():
                self.trainer.standard_module_loss_weight = 1

        return next_obs

    ##################################################
    # Training
    ##################################################
    def train_ir_agent(self):
        self.ir_agent.train()
        next_observations, observations, last_actions, last_terminations, last_exists = \
            self.rb.last_samples(self.config["train_ir_freq"])

        batch_observations = self._preprocess(observations)
        batch_observations = batch_observations.reshape(
            -1, *self.envs.single_observation_space.shape)

        inds = np.arange(batch_observations.shape[0])
        for _ in range(self.train_intrinsic_intensity):
            np.random.shuffle(inds)
            ir_train_info = self.trainer.on_policy_update_ir_agent(
                batch_observations[inds])

        self.ir_agent.eval()
        return ir_train_info

    def train_model(self):
        beta = linear_schedule(self.n_transitions,
                               init=self.buffer_init_beta,
                               max_n=self.total_timesteps)

        for _ in range(self.training_intensity):
            data = rb.sample(self.batch_size, beta=beta)
            train_info = self.trainer.train_once(data)
            self.rb.update_priorities(
                data.batch_inds, data.env_inds,
                train_info["q_network"]["abs_td_error"].numpy())

        train_info = {**train_info,
                      "n_transitions": self.n_transitions}
        return train_info

    def save_model(self, global_step, optimizers):
        q_optimizer = optimizers["q_optimizer"]
        actor_optimizer = optimizers["actor_optimizer"]
        a_optimizer = optimizers["a_optimizer"]
        ir_optimizer = optimizers["ir_optimizer"]

        checkpoint_path = path.join(
            self.result_dir, f"n_samples_{self.n_transitions:07d}")

        save_dict = {
            'global_step': global_step,
            'num_envs': self.num_envs,
            'num_transitions': self.n_transitions,
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': {
                "q_network": q_optimizer.state_dict()},
            "log_alpha": self.trainer.log_alpha}

        if actor_optimizer is not None:
            save_dict["optimizer_state_dict"]["actor"] = actor_optimizer.state_dict()

        if self.with_int_rewards:
            save_dict["ir_model_state_dict"] = self.ir_agent.state_dict()
            save_dict["optimizer_state_dict"]["ir_optimizer"] = \
                ir_optimizer.state_dict()

        torch.save(save_dict, checkpoint_path)

        buffer_path = path.join(
            self.result_dir, f"buffer_{self.n_transitions:08d}.pkl")
        self.rb.save(buffer_path)
        print(f"Save checkpoint in {checkpoint_path}")

    ##################################################
    # Evaluate
    ##################################################
    def _estimate_optimal_actions(self, obs, deterministic=False):
        if deterministic:
            if self.use_standard_module_eval is None:
                actions = self.agent.get_action(self._preprocess(obs), with_exploration=False)
            else:
                # Use the standard_module because it only consider extrinsic rewards (task)
                actions = self.agent.get_action(
                    self._preprocess(obs), use_standard_module=1, with_exploration=False)
        else:
            if self.use_standard_module_eval is None:
                actions, _, _ = self.agent.get_action(self._preprocess(obs))
            else:
                actions, _, _ = self.agent.get_action(
                    self._preprocess(obs), use_standard_module=self.use_standard_module_eval)

        actions = actions.detach().cpu().numpy()
        return actions

    @torch.no_grad()
    def evaluate(self):
        self.agent.eval()
        episode_returns = []
        envs.training = False
        obs = self.val_envs.reset()

        while len(episode_returns) < self.num_eval_episodes:
            actions = self._estimate_optimal_actions(obs, deterministic=True)
            next_obs, rewards, dones, infos = self.val_envs.step(actions)

            if dones.any():
                for done, info in zip(dones, infos):
                    if done:
                        episode_returns.append(info["episode"]['r'])
            obs = next_obs
        envs.training = True
        self.agent.train()
        return episode_returns

    ##################################################
    # Conditions
    ##################################################
    def is_learning_starts(self, global_step):
        if global_step >= self.num_samples_before_learning:
            return True
        return False

    def is_report_rollout(self, global_step):
        if global_step % self.report_rollout_freq == 0:
            return True
        return False

    def is_train_ir_agent(self, global_step):
        return global_step % self.train_ir_freq == 0

    def is_train_model(self, global_step):
        return (self.is_learning_starts(global_step) and
                global_step % self.train_freq == 0)

    def is_report_train(self, global_step):
        if (self.is_learning_starts(global_step) and
            global_step % self.report_loss_freq == 0):
            return True
        return False

    def is_save_model(self, global_step):
        return (global_step % self.save_model_freq == 0)

    def is_evaluate_model(self, global_step):
        if (self.is_learning_starts(global_step) and
            global_step % self.eval_freq == 0):
            return True
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # System setup
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--use_tensorboard", default=0, type=int)
    parser.add_argument("--n_envs", default=4, type=int)
    parser.add_argument("--n_eval_envs", default=2, type=int)
    parser.add_argument("-Q", "--quiet", action="store_true")
    parser.add_argument("--use_cuda", default=1, type=int)
    # Experiment setup
    parser.add_argument("--exp_name", default="exp_911_DS20_rnd_sac")
    parser.add_argument("-ei", "--exp_index", default="e000")
    parser.add_argument("--basic_config", default="", type=str,
                        help="Forcing assign a basic_config file.")
    args = parser.parse_args()
    args.quiet = True

    result_dir = f"./results/{args.exp_name}/{args.exp_index}"
    os.makedirs(result_dir, exist_ok=True)

    config = load_and_parse_configs(
        exp_name=args.exp_name,
        path=f"./configs/{args.exp_name[:5]}/{args.exp_name}.yaml",
        result_dir=result_dir,
        basic_config_name=args.basic_config)

    writer = None
    if args.use_tensorboard:
        writer = SummaryWriter(f"results/logs/{args.exp_name[:5]}/{args.exp_name}/{args.exp_index}-{int(time.perf_counter())}")

    device = torch.device("cpu")
    if args.use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = set_random_seed(args.seed)

    render_mode = "rgb_array"
    init_step = 1

    ##################################################
    # Environment
    ##################################################
    print(">>> Build environment")
    num_envs = args.n_envs
    num_eval_envs = args.n_eval_envs

    env_id = config.get("env_id", "grid_maze_env")

    wrapper_list = [gym.wrappers.FlattenObservation,
                    gym.wrappers.RecordEpisodeStatistics]

    print(f">>> Buile DeepMind Behavior Suite Environment: {env_id}")
    kwargs = {}
    seeds = [seed for i in range(num_envs)]
    env_list = [make_env(env_id, seeds[i], kwargs, wrapper_list)
                for i in range(num_envs)]
    val_seeds = [seed for i in range(num_eval_envs)]
    val_env_list = [make_env(env_id, val_seeds[i], kwargs, wrapper_list)
                    for i in range(num_eval_envs)]

    envs = gym.vector.AsyncVectorEnv(env_list)
    val_envs = gym.vector.AsyncVectorEnv(val_env_list)

    ##################################################
    # Agent
    ##################################################
    print(">>> Build agent")
    gamma = config["gamma"]
    gae_lambda = config.get("gae_lambda", 0.95)

    is_discrete_action = config["action_type"] == "discrete"
    if config["rl_agent"] == "KEA-SAC":
        if is_discrete_action:
            agent = DiscreteKEA(envs.single_observation_space.shape,
                                envs.single_action_space.n,
                                config["backbone_type"],
                                device,
                                latent_dim=config.get("latent_dim", 256))
            print(">>> Build a DiscreteKEA")

    target_entropy, log_alpha, log_alpha_co = None, None, None
    if "SAC" in config["rl_agent"]:
        if config.get("autotune", False):
            target_entropy = -torch.prod(torch.Tensor(
                envs.single_action_space.shape).to(device)).item()
            print(f"Target Entropy (original): {target_entropy}")
            if config.get("target_entropy", None):
                target_entropy = config["target_entropy"]
            print(f"Target Entropy (assigned): {target_entropy}")
            log_alpha = torch.zeros(1, requires_grad=True, device=device)
            alpha = log_alpha.exp().item()

        else:
            entropy_alpha = config.get("entropy_alpha", 0.2)
            log_alpha = torch.tensor(np.log([entropy_alpha]), device=device)

        entropy_alpha = config.get("entropy_alpha", 0.2)
        log_alpha_co = torch.tensor(np.log([entropy_alpha]), device=device)

    ir_agent = None
    if config["with_intrinsic_rewards"]:
        if config["IR_model"] == "RND":
            ir_agent = RandomNetworkDistillation(
                envs.single_observation_space.shape,
                config["ir_feat_dim"],
                config["backbone_type"],
                device,
                norm_type=config.get("norm_type", "run_mean_std"),
                with_extra_obs=config.get("with_extra_obs", False))
            print(">>> Build RND")

        if config.get("ir_clip", None) is not None:
            ir_agent.ir_clip = config.get("ir_clip")
            print(f"Change IR clip: {ir_agent.ir_clip}")

    ##################################################
    # Optimizer
    ##################################################
    print(">>> Build optimizers")
    if config["rl_agent"] == "KEA-SAC":
        q_optimizer = optim.Adam(
            list(agent.standard_module.backbone.parameters()) +
            list(agent.standard_module.qf1.parameters()) +
            list(agent.standard_module.qf2.parameters()) +
            list(agent.novel_aug_module.backbone.parameters()) +
            list(agent.novel_aug_module.qf1.parameters()) +
            list(agent.novel_aug_module.qf2.parameters()),
            lr=config["critic_learning_rate"])
        actor_optimizer = optim.Adam(
            list(agent.standard_module.backbone.parameters()) +
            list(agent.standard_module.actor.parameters()) +
            list(agent.novel_aug_module.backbone.parameters()) +
            list(agent.novel_aug_module.actor.parameters()),
            lr=config["actor_learning_rate"])

    a_optimizer = None
    if config.get("autotune", False):
        a_optimizer = optim.Adam([log_alpha], lr=config["critic_learning_rate"])

    # Intrinsic reward model optimizer
    ir_optimizer = None
    if config["with_intrinsic_rewards"]:
        ir_optimizer = optim.Adam(
            list(ir_agent.backbone.parameters()) +
            list(ir_agent.predictor.parameters()),
            lr=config["ir_learning_rate"])

    optimizers = {
        "q_optimizer": q_optimizer,
        "actor_optimizer": actor_optimizer,
        "a_optimizer": a_optimizer,
        "ir_optimizer": ir_optimizer}

    ##################################################
    # Trainer
    ##################################################
    print(f">>> Build trainer: {config['rl_agent']}")
    if config["rl_agent"] == "KEA-SAC":
        if is_discrete_action:
            trainer = DiscreteKEASACTrainer(
                agent, ir_agent, optimizers, log_alpha, target_entropy, device,
                result_dir, config, log_alpha_co=log_alpha_co)

    reporter = Reporter(total_int_reward=[], num_use_novel_aug_module=[],
                        epsilon_greedy=[], visited_ratio=[])

    ##################################################
    # Replay Buffer
    ##################################################
    dummy_action_space = gym.spaces.Box(
        low=-1, high=1, shape=envs.single_action_space.shape)
    dummy_observation_space = gym.spaces.Box(
        low=-1, high=1, shape=envs.single_observation_space.shape)
    print(">>> Build buffer")
    rb = PrioritizedReplayBuffer(
        config["buffer_size"],
        dummy_observation_space,
        dummy_action_space,
        device,
        n_envs=envs.num_envs,
        alpha=config["buffer_alpha"])

    ##################################################
    # Training
    ##################################################
    runner = Runner(envs, agent, ir_agent, trainer, rb, reporter, config,
                    result_dir, device, num_envs, seeds=seeds,
                    num_eval_envs=num_eval_envs, val_envs=val_envs,
                    render_mode=render_mode)

    obs = envs.reset()

    train_info, train_ir_info = {}, {}
    print(f">>> Training Start! (with seed: {seed})")
    for global_step in range(init_step, runner.total_timesteps + 1):
        if config["decay_lr"]:
            decay_lr(optimizers, global_step, runner.total_timesteps, config)

        next_obs = runner.step(obs, global_step)
        obs = next_obs

        # Update IR agent
        if config["with_intrinsic_rewards"] and runner.is_train_ir_agent(global_step):
            train_ir_info = runner.train_ir_agent()

        # Update model
        if runner.is_train_model(global_step):
            train_info = runner.train_model()

        elif trainer.minibatch_pos >= trainer.capacity:
            trainer.minibatch_pos = 0
            batch_ext_returns, advantages = trainer.compute_return_and_advantage(
                runner._preprocess(next_obs))
            trainer.prepare_mb_inputs(batch_ext_returns, advantages, num_envs, envs)

            capacity_inds = np.arange(trainer.capacity * num_envs)
            if config["with_intrinsic_rewards"]:
                train_ir_info = runner.train_ir_agent()

            for epoch in range(config["training_intensity"]):
                np.random.shuffle(capacity_inds)
                for start in range(0, len(capacity_inds), config["batch_size"]):
                    end = start + config["batch_size"]
                    batch_inds = capacity_inds[start:end]

                    train_info = trainer.update_model(batch_inds)

            train_info = {"q_network": train_info,
                          "actor": {},
                          "recomputed_int_reward": -1,
                          "n_transitions": runner.n_transitions}

        # Report rollout info
        if runner.is_report_rollout(global_step):
            if env_id == "grid_maze_env":
                num_total_states = np.prod(envs.get_attr("grid_size")[0])
                num_visited_states = len(np.unique(
                    rb.observations.reshape(-1, *rb.obs_shape), axis=0))
                visited_ratio = num_visited_states / num_total_states
                reporter.visited_ratio.append(visited_ratio)
            reporter.report_rollout_info(writer, runner.n_transitions)

        # Report loss info
        if runner.is_report_train(global_step):
            info = {**train_info, "IR": train_ir_info}
            reporter.report_train_info(writer, info)

        # Save model
        if runner.is_save_model(global_step):
            runner.save_model(global_step, optimizers)

        # Evaluate
        if runner.is_evaluate_model(global_step):
            episode_returns = runner.evaluate()
            avg_episode_return = np.mean(episode_returns)
            max_return, min_return = np.max(episode_returns), np.min(episode_returns)
            print(f"[{args.exp_index}] [{runner.n_transitions:6d}] Eval Return: {avg_episode_return:.4f} ({max_return:.4f}, {min_return:.4f})")
            if writer:
                writer.add_scalar("eval/total_return", avg_episode_return, runner.n_transitions)
                writer.add_scalar("eval/min_total_return", min_return, runner.n_transitions)
                writer.add_scalar("eval/max_total_return", max_return, runner.n_transitions)

    runner.save_model(global_step, optimizers)
    envs.close()
    if writer:
        writer.close()
