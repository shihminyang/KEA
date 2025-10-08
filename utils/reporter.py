import numpy as np


class Reporter:
    # [code release] Used
    def __init__(self, **kwargs):
        self.__dict__ = {}
        self.total_return = []
        self.total_reward = []
        self.episode_length = []
        self.entropy = []

        self.hist_complete = []

        self.__dict__.update(kwargs)

    def __getitem__(self, key):
        return self.__dict__.get(key)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __repr__(self):
        return repr(self.__dict__)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    ##################################################
    # Report
    ##################################################
    def _reset(self):
        for k in self.keys():
            self[k] = self[k][-1:]
        #     self[k] = []

    def report_rollout_info(self, writer, n_transitions):
        if len(self.total_reward) == 0:
            return

        msg1 = f"Total Return: {np.mean(self.total_return): 8.4f}"
        msg2 = f"Total Reward: {np.mean(self.total_reward): 8.4f}"
        msg3 = ""
        if len(self.hist_complete) > 0:
            msg3 = f"Complete Rate: {np.mean(self.hist_complete): 8.4f}"
        print(f"[{n_transitions:6d}] Episode {msg1} | {msg2} | {msg3}")

        if writer:
            for k, v in self.items():
                if len(v) > 0:
                    writer.add_scalar(f"rollout/{k}", np.mean(v), n_transitions)

        self._reset()

    def report_train_info(self, writer, info):
        if len(info) == 0:
            return
        n = info["n_transitions"]

        if writer is None:
            return

        for k, v in info["q_network"].items():
            if "td_error" in k:
                v = v.mean()
            writer.add_scalar(f"train/Q/{k}", v, n)

        for k, v in info["actor"].items():
            writer.add_scalar(f"train/Actor/{k}", v, n)

        for k, v in info.get("phi", {}).items():
            writer.add_scalar(f"train/phi/{k}", v, n)

        for k, v in info.get("IR", {}).items():
            writer.add_scalar(f"train/IR/{k}", v, n)

        if "recomputed_reward" in info:
            writer.add_scalar("train/reward", info["recomputed_reward"], n)

        if "recomputed_int_reward" in info:
            writer.add_scalar("train/int_reward", info["recomputed_int_reward"], n)
