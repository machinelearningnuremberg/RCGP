import json
import os
import sys
import pickle
import numpy as np
from autorlbench_utils import get_metrics
# 'Bowling-v0'
# ENVIRONMENTS = ['Pong-v0', 'Alien-v0', 'BankHeist-v0',
#                 'BeamRider-v0', 'Breakout-v0', 'Enduro-v0',
#                 'Phoenix-v0', 'Seaquest-v0', 'SpaceInvaders-v0',
#                 'Riverraid-v0', 'Tennis-v0', 'Skiing-v0', 'Boxing-v0',
#                 'Bowling-v0', 'Asteroids-v0',
#                 'Ant-v2', 'Hopper-v2', 'Humanoid-v2',
#                 'CartPole-v1', 'MountainCar-v0', 'Acrobot-v1', 'Pendulum-v0']
# ALGORITHMS = ["PPO", "DDPG", "A2C"]
# SEED = 2
# with open('config_space_ppo', 'rb') as f:
#     config_space_ppo = pickle.load(f)
# with open('config_space_ddpg', 'rb') as f:
#     config_space_ddpg = pickle.load(f)
# with open('config_space_ppo', 'rb') as f:
#     config_space_a2c = pickle.load(f)
# CONFIG_SPACES = [config_space_ppo, config_space_ddpg, config_space_a2c]
# for i, algorithm in enumerate(ALGORITHMS):
#     for environment in ENVIRONMENTS:
#         print(environment)
#         for config in CONFIG_SPACES[i]:
#             if i == 0:
#                 with open(os.path.join('data_arl_bench', algorithm, environment,
#                                        '%s_%s_random_lr_%s_gamma_%s_clip_%s_seed%s_eval.json'
#                                        % (environment, algorithm, config[0], config[1], config[2], SEED))) as f:
#                     data = json.load(f)
#                 if np.any(np.isnan(data["timesteps_eval"])):
#                     timesteps_eval = []
#                     for timestep in range(200):
#                         timesteps_eval.append(10000*(timestep+1))
#                     data["timesteps_eval"] = timesteps_eval
#                     data["learning_rate"] = int(data["learning_rate"])
#                     print(len(timesteps_eval))
#                     with open(os.path.join('data_arl_bench', algorithm, environment,
#                                            '%s_%s_random_lr_%s_gamma_%s_clip_%s_seed%s_eval.json'
#                                            % (environment, algorithm, config[0], config[1], config[2], SEED)), 'w+') as f:
#                         json.dump(data, f)
#             elif i == 2:
#                 with open(os.path.join('data_arl_bench', algorithm, environment,
#                                        '%s_%s_random_lr_%s_gamma_%s_seed%s_eval.json'
#                                        % (environment, algorithm, config[0], config[1], SEED))) as f:
#                     data = json.load(f)
#                 if np.any(np.isnan(data["timesteps_eval"])):
#                     timesteps_eval = []
#                     for timestep in range(200):
#                         timesteps_eval.append(10000 * (j + 1))
#                     data["timesteps_eval"] = timesteps_eval
#                     data["learning_rate"] = int(data["learning_rate"])
#                     print(len(timesteps_eval))
#                     with open(os.path.join('data_arl_bench', algorithm, environment,
#                                            '%s_%s_random_lr_%s_gamma_%s_seed%s_eval.json'
#                                            % (environment, algorithm, config[0], config[1], SEED)), 'w+') as f:
#                         json.dump(data, f)
# array = [-717.3554898, -1112.8654218, 154.526738, -899.1747819, -233.5897081, -545.7575378, -400.0553614, -1178.3692449999999, -1780.0769999, -306.8557404, -472.77993070000014, -221.9450645, -111.8235877, -65.7607945, -141.3992232, -12.743249299999999, -195.9629399, -136.20864680000003, -124.85891799999999, -310.1289373, -160.4623705, -405.5920672, -588.1510425, -1082.7569268000002, -2666.9897708999997, -2958.3959815000003, -2988.3652575000006, -2903.6146878, -2831.92252, -2927.3325849999997, -2996.0922418, -2997.3483661, -2997.5782217999995, -2997.2491032, -2998.2092196999997, -2997.585088, -2997.4107414, -2998.020329, -2997.7634058, -2998.1517409, -2996.8040885, -2998.4286502, -2997.288824, -2998.5225145000004, -2996.8491495000007, -2998.078501, -2998.0717154, -2997.8157446, -2997.6368208999997, -2997.0240635, -2998.1896810000003, -2997.8217202, -2998.2890146, -2998.4701321, -2997.1701916, -2997.9563706999998, -2997.3241067, -2997.5312021, -2997.9651506, -2996.7977010000004, -2996.9505922, -2996.0714264, -2997.7676445, -2998.4095933, -2998.2521758, -2997.145991, -2997.6415792999996, -2996.3201729, -2997.5225607, -2996.7699470000002, -2997.8250705999994, -2997.8611441000003, -2997.1081185000007, -2997.5879465, -2998.127807, -2997.5527571000002, -2997.476218, -2996.767583, -2996.9599928, -2997.0237022, -2997.336727, -2997.6583979, -2996.4756296000005, -2996.9233481, -2996.6839466000006, -2996.4113371999997, -2914.5928184000004, -2976.275803, -2997.0553492000004, -2994.5781819, -2993.9445771, -2969.6610017, -2992.417413, -2997.2863489, -2981.8978153999997, -2990.3413973, -2981.3700925, -2961.6687843, -2986.6109745, -2993.0004833]
# print(len(array))
ENVIRONMENTS = ['Pong-v0', 'Alien-v0', 'BankHeist-v0',
                'BeamRider-v0', 'Breakout-v0', 'Enduro-v0',
                'Phoenix-v0', 'Seaquest-v0', 'SpaceInvaders-v0',
                'Riverraid-v0', 'Tennis-v0', 'Skiing-v0', 'Boxing-v0',
                'Bowling-v0', 'Asteroids-v0',
                'Ant-v2', 'Hopper-v2', 'Humanoid-v2',
                'CartPole-v1', 'MountainCar-v0', 'Acrobot-v1', 'Pendulum-v0']
ENVIRONMENTS_DDPG = ['Ant-v2', 'Hopper-v2', 'Humanoid-v2']
ALGORITHMS = ["DDPG", "TD3", "SAC"]  #["DDPG"]  #
# ALGORITHMS = ["PPO", "A2C"]
SEEDS = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
for seed in SEEDS:
    for environment in ENVIRONMENTS_DDPG:
        print(environment)
        for algorithm in ALGORITHMS:
            all_data = []
            with open(os.path.join('runs_raw', 'LCGP','LCGP_%s_%s_seed%s.json' % (algorithm, environment, seed))) as f:
                incumbents = []
                wallclock_time_eval = []
                timesteps_eval = []
                last_timesteps_eval = 0
                last_wallclock_time_eval = 0
                inc_budget = 100
                inc = -np.inf
                for i, obj in enumerate(f):
                    data = json.loads(obj)

                    budget = data["budget"]
                    if algorithm == "PPO":
                        config = {"lr": data["config"][0], "gamma": data["config"][1], "clip": data["config"][2]}
                    elif algorithm in ["DDPG", "SAC", "TD3"]:
                        config = {"lr": data["config"][0], "gamma": data["config"][1], "tau": data["config"][2]}
                    elif algorithm == "A2C":
                        config = {"lr": data["config"][0], "gamma": data["config"][1]}
                    result = get_metrics(search_space=algorithm, environment=environment, config=config,
                                         budget=budget,
                                         seed=0)
                    if len(wallclock_time_eval) > 0:
                        last_wallclock_time_eval = wallclock_time_eval[-1]
                        last_timesteps_eval = timesteps_eval[-1]
                    if data["budget"] > inc_budget:
                        inc_budget = data["budget"]
                        inc = result["eval_avg_returns"][-1]
                    elif data["budget"] <= inc_budget:
                        if data["incumbent"] > inc:
                            inc = result["eval_avg_returns"][-1]
                    incumbents.append(inc)
                    wallclock_time_eval.append(result["eval_timestamps"][budget-1] + last_wallclock_time_eval)
                    timesteps_eval.append(result["eval_timesteps"][budget-1] + last_timesteps_eval)

                final = {"incumbents":incumbents, "wallclock_time_eval": wallclock_time_eval, "timesteps_eval": timesteps_eval}
                with open(os.path.join('data_LCGP', 'LCGP_seed%s_%s_%s.json' % (seed, environment, algorithm)), 'w+') as f:
                    json.dump(final, f)

