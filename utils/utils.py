import numpy as np
import yaml
import torch.nn as nn
import time


def load_and_parse_configs(exp_name, path, result_dir, basic_config_name=""):
    # [code release] Used
    config = load_config(path)
    if basic_config_name == "":
        basic_config_name = config["basic_config"]
    print("Using basic config: ", basic_config_name)
    time.sleep(1)

    basic_config = load_config(f"./configs/{exp_name[:5]}/{basic_config_name}")

    config = {**basic_config, **config}

    # Save config and basic_config
    with open(f"{result_dir}/{exp_name}.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    with open(f"{result_dir}/{basic_config_name}", 'w') as f:
        yaml.dump(basic_config, f, default_flow_style=False)

    return config


def load_config(config_path):
    """ Loading config file. """

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


########################################
# Network initialization
########################################
def initialize_weights_orthogonal(net: nn.Module, std=np.sqrt(2), bias_const=0.0) -> None:
    # [code release] Used
    assert isinstance(net, nn.Module)

    for layer in net.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(layer.weight, std)
            nn.init.constant_(layer.bias, bias_const)

########################################
# Normalize
########################################
class RunningMeanStd(object):
    """
    Implemented based on:
    - https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    - https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/mpi_util.py#L179-L214
    - https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/running_mean_std.py
    """
    # [code release] Used
    def __init__(self, epsilon=1e-4, momentum=None, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
        self.eps = epsilon
        self.momentum = momentum

    def clear(self):
        self.__init__(self.eps, self.momentum)

    @staticmethod
    def update_ema(old_data, new_data, momentum):
        if old_data is None:
            return new_data
        return old_data * momentum + new_data * (1.0 - momentum)

    def update(self, x):
        batch_mean, batch_std, batch_count = np.mean(x, axis=0), np.std(x, axis=0), x.shape[0]
        batch_var = np.square(batch_std)
        if self.momentum is None or self.momentum < 0:
            self.update_from_moments(batch_mean, batch_var, batch_count)
        else:
            self.mean = self.update_ema(self.mean, batch_mean, self.momentum)
            new_var = np.mean(np.square(x - self.mean))
            self.var = self.update_ema(self.var, new_var, self.momentum)
            self.std = np.sqrt(self.var)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count
        self.mean = new_mean
        self.var = new_var
        self.std = np.sqrt(new_var)
        self.count = new_count
