"""
Worker for Examples 1-4
=======================

This class implements a very simple worker used in the firt examples.
"""

import numpy
import time
import json
from ConfigSpace import ConfigurationSpace as CS
from ConfigSpace import UniformIntegerHyperparameter, CategoricalHyperparameter
from hpbandster.core.worker import Worker
from autorlbench_utils import get_metrics



class MyWorker(Worker):

    def __init__(self, *args, ALGORITHM, ENVIRONMENT, SEED, **kwargs):
        super().__init__(*args, **kwargs)

        self.ALGORITHM = ALGORITHM
        self.ENVIRONMENT = ENVIRONMENT
        self.SEED = SEED

    def compute(self, config, budget, **kwargs):
        budget = int(budget)
        cfg = []
        learning_rate = config["lr"]
        cfg.append(learning_rate)
        gamma = config["gamma"]
        cfg.append(gamma)
        if self.ALGORITHM == "PPO":
            clip = config["clip"]
            cfg.append(clip)
        elif self.ALGORITHM == "DDPG":
            clip = config["tau"]
            cfg.append(clip)

        result = get_metrics(search_space=self.ALGORITHM, environment=self.ENVIRONMENT, config=config, seed=self.SEED, budget=100)
        eval_reward = result["eval_avg_returns"][budget-1]
        eval_timestamp = result["eval_timestamps"][budget-1]
        eval_timesteps = result["eval_timesteps"][budget-1]
        final_eval_reward = result["eval_avg_returns"][-1]
        final_eval_timestamp = result["eval_timestamps"][-1]
        final_eval_timesteps = result["eval_timesteps"][-1]
        data = {"config": cfg,
                "eval_return": eval_reward, "eval_timestamp": eval_timestamp, "eval_timesteps": eval_timesteps,
                "full_avg_return": eval_reward, "full_eval_timestamp": eval_timestamp, "full_eval_timesteps": eval_timesteps,
                "budget": budget}
        with open("BOHB_%s_%s_seed%s_eval.json" % (self.ALGORITHM, self.ENVIRONMENT, self.SEED), 'a+') as f:
            json.dump(data, f)
            f.write("\n")

        return ({
                'loss': -1 * eval_reward,  # this is the a mandatory field to run hyperband
                'info': data  # can be used for any user-defined information - also mandatory
        })

    @staticmethod
    def get_configspace(ALGORITHM):
        cs = CS()
        learning_rate = UniformIntegerHyperparameter(
            "lr", -6, -2, default_value=-4
        )
        gamma = CategoricalHyperparameter(
            "gamma", choices=[0.8, 0.9, 0.95, 0.98, 0.99, 1.0]
        )
        cs.add_hyperparameters(
            [
                learning_rate,
                gamma
            ]
        )
        if ALGORITHM == "PPO":
            clipping = CategoricalHyperparameter(
                "clip", choices=[0.2, 0.3, 0.4]
            )
            cs.add_hyperparameters([clipping])
        elif ALGORITHM == "DDPG":
            tau = CategoricalHyperparameter(
                "tau", choices=[0.001, 0.005, 0.01]
            )
            cs.add_hyperparameters([tau])
        return (cs)
