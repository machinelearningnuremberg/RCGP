import logging
from autorlbench_utils import get_metrics
logging.basicConfig(level=logging.INFO)

import warnings
import numpy as np
from ConfigSpace import ConfigurationSpace
import ConfigSpace as CS
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


import os
import sys
import gym
import numpy as np
import json
from DEHB.dehb import DEHB

SEED = int(sys.argv[3])
ALGORITHMS = ["PPO", "DDPG", "A2C"]
ALGORITHM = ALGORITHMS[int(sys.argv[1])]
ENVS = ['Pong-v0', 'Alien-v0', 'BankHeist-v0',
        'BeamRider-v0', 'Breakout-v0', 'Enduro-v0',
        'Phoenix-v0', 'Seaquest-v0', 'SpaceInvaders-v0',
        'Riverraid-v0', 'Tennis-v0', 'Skiing-v0', 'Boxing-v0',
        'Bowling-v0', 'Asteroids-v0',
        'Ant-v2', 'Hopper-v2', 'Humanoid-v2',
        'CartPole-v1', 'MountainCar-v0', 'Acrobot-v1', 'Pendulum-v0']
ENVIRONMENT = ENVS[int(sys.argv[2])]

# Target Algorithm
def query_arlbench(cfg, budget):
    budget = int(budget)
    config = []
    learning_rate = cfg["lr"]
    config.append(learning_rate)
    gamma = cfg["gamma"]
    config.append(gamma)
    if ALGORITHM == "PPO":
        clip = cfg["clip"]
        config.append(clip)
    elif ALGORITHM == "DDPG":
        clip = cfg["tau"]
        config.append(clip)

    result = get_metrics(search_space=ALGORITHM, environment=ENVIRONMENT, config=cfg, seed=SEED, budget=budget)
    eval_reward = result["eval_avg_returns"][-1]
    eval_timestamp = result["eval_timestamps"][-1]
    eval_timesteps = result["eval_timesteps"][-1]

    data = {"config": config,
            "eval_return": eval_reward, "eval_timestamp": eval_timestamp, "eval_timesteps": eval_timesteps,
            "budget": budget}
    with open("DEHB_%s_%s_seed%s_eval.json" % (ALGORITHM, ENVIRONMENT, SEED), 'a+') as f:
        json.dump(data, f)
        f.write("\n")
    return {
        "fitness": -1 * eval_reward,
        "cost": budget
    }


if __name__ == "__main__":
    # Build Configuration Space which defines all parameters and their ranges.
    # To illustrate different parameter types,
    # we use continuous, integer and categorical parameters.
    cs = ConfigurationSpace()
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
    dimensions = len(cs.get_hyperparameters())

    # Max budget for hyperband can be anything. Here, we set it to maximum no. of epochs to train the MLP for
    max_epochs = 100

    # Intensifier parameters
    intensifier_kwargs = {"initial_budget": 5, "max_budget": max_epochs, "eta": 3}
    # seed = int(sys.argv[1])
    # To optimize, we pass the function to the SMAC-object
    smac = DEHB(
        dimensions=dimensions,
        n_workers=1,
        f=query_arlbench,
        cs=cs,
        min_budget=5,
        max_budget=100,
        output_path='./temp'
    )
    _,_,_ = smac.run(fevals=200, verbose=False, save_intermediate=True)
    print(smac.vector_to_configspace(smac.inc_config))