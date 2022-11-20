#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import argparse
import time
import os
from DeepKernelGPHelpers import regret,EI
from sklearn.preprocessing import MinMaxScaler
from DeepKernelGPModules import ExactGPLayer
import torch
from scipy.stats import norm
import os
import gym
import numpy as np
import json
import pickle
from autorlbench_utils import get_metrics
import gpytorch
import logging

ENVS = ['Pong-v0', 'Alien-v0', 'BankHeist-v0',
        'BeamRider-v0', 'Breakout-v0', 'Enduro-v0',
        'Phoenix-v0', 'Seaquest-v0', 'SpaceInvaders-v0',
        'Riverraid-v0', 'Tennis-v0', 'Skiing-v0', 'Boxing-v0',
        'Bowling-v0', 'Asteroids-v0',
        'Ant-v2', 'Hopper-v2', 'Humanoid-v2',
        'CartPole-v1', 'MountainCar-v0', 'Acrobot-v1', 'Pendulum-v0']

def EI(incumbent, model, likelihood, support_x, support_y, query, budget, inc_x, config):
    q_tuple = (query[i] for i in range(len(query)))
    if inc_x == q_tuple:
        return -np.inf
    if q_tuple not in support_x:
        x_query = np.array(query)
        mu, stddev = predict(model, likelihood, support_x, support_y, x_query, config=config)
        mu = mu.reshape(-1, )
        stddev = stddev.reshape(-1, )
        with np.errstate(divide='warn'):
            imp = mu - incumbent
            Z = imp / stddev
            score = imp * norm.cdf(Z) + stddev * norm.pdf(Z)
            return score.item()
    # elif len(support_y[q_tuple]) < budget:
    #     x_query = np.array(query)
    #     mu, stddev = model.predict(support_x, support_y, x_query)
    #     mu = mu.reshape(-1, )
    #     stddev = stddev.reshape(-1, )
    #     with np.errstate(divide='warn'):
    #         imp = mu - incumbent
    #         Z = imp / stddev
    #         score = imp * norm.cdf(Z) + stddev * norm.pdf(Z)
    #         return score.item()
    return -np.inf

def propose_location(incumbent, X_sample, Y_sample, gpr, likelihood, budget, dim, config_space, inc_x, max_Y, conf):
    '''
    Proposes the next sampling point by optimizing the acquisition function.

    Args:
        acquisition: Acquisition function.
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.

    Returns:
        Location of the acquisition function maximum.
    '''
    scores = []
    for x in config_space:
        # if x not in X_sample:
        config = list(x)
        if x not in X_sample:
            acq = EI(incumbent/max_Y, gpr, likelihood, support_x=X_sample, support_y=Y_sample, query=config, budget=budget, inc_x=inc_x, config=conf)
        else:
            acq = -np.inf
        # else:
        # acq = np.array([-np.inf])
        scores.append(acq)
    min_x = np.amax(scores)
    print("Min x: %s" % min_x)
    cands = []
    idxs = []
    for i, score in enumerate(scores):
        if score == min_x:
            cands.append(10)
            idxs.append(i)
        else:
            cands.append(0)
    if sum(cands) > 10:
        min_x = np.random.choice(idxs)
    else:
        min_x = np.argmax(scores)
    return min_x

def find_incumbent(y, budget):
    max_budget = np.amax([len(rs) for rs in y.values()])

    if max_budget >= budget:
        incumbent = -np.inf
        inc_x = None
        inc_budget = budget
        for config, rs in y.items():
            if len(rs) >= budget:
                if rs[budget-1] > incumbent:
                    incumbent = rs[budget-1]
                    inc_x = config
    else:
        incumbent = -np.inf
        inc_x = None
        inc_budget = max_budget
        for config, rs in y.items():
            if rs[-1] > incumbent:
                incumbent = rs[-1]
                inc_x = config
                inc_budget = len(rs)
    return incumbent, inc_x, inc_budget

def get_model_likelihood_mll(train_size, in_features, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x = torch.ones(train_size, in_features).to(device)
    train_y = torch.ones(train_size).to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, config=config,
                         dims=in_features)
    model = model.to(device)
    likelihood = likelihood.to(device)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(device)
    return model, likelihood, mll

def train(model, likelihood, mll, support_x, support_y, optimizer, epochs=1000, verbose=False, config=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs, labels = prepare_data(support_x, support_y)
        X = inputs
        max_Y = max(labels)
        if max_Y == 0: max_Y =1
        labels /= max_Y
        Y = labels
        inputs, labels = torch.tensor(inputs, dtype=torch.float).to(device), torch.tensor(labels, dtype=torch.float).to(device)
        labels = labels/max(labels)
        losses = [np.inf]
        best_loss = np.inf
        starttime = time.time()
        patience = 0
        max_patience = config["patience"]
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            model.set_train_data(inputs=inputs, targets=labels, strict=False)
            predictions = model(inputs)
            try:
                loss = -mll(predictions, model.train_targets)
                loss.backward()
                optimizer.step()
            except Exception as ada:
                logging.info(f"Exception {ada}")
                break

            if verbose:
                print("Iter {iter}/{epochs} - Loss: {loss:.5f}   noise: {noise:.5f}".format(
                    iter=_ + 1, epochs=epochs, loss=loss.item(),
                    noise=likelihood.noise.item()))
            losses.append(loss.detach().to("cpu").item())
            if best_loss > losses[-1]:
                best_loss = losses[-1]
            # if np.allclose(losses[-1], losses[-2], atol=self.config["loss_tol"]):
            #     patience += 1
            # else:
            #     patience = 0
            # if patience > max_patience:
            #     break
        logging.info(
            f"Current Iteration: {len(Y)} | Incumbent {max(Y)} | Duration {np.round(time.time() - starttime)} | Epochs {_} | Noise {likelihood.noise.item()}")
        return model, likelihood, mll, max_Y

def prepare_data(support_x, support_y):
    inputs = []
    labels = []
    for cfg in support_y.keys():
        lc = support_y[cfg]
        inputs.append(np.array(list(cfg)))
        labels.append(lc[-1])
    return np.array(inputs), np.array(labels)

def predict(model, likelihood, support_x, support_y, query_x,  noise_fn=None, config=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs, labels = prepare_data(support_x, support_y)
        X = inputs
        max_Y = max(labels)
        if max_Y == 0: max_Y = 1
        labels /= max(labels)
        Y = labels/max_Y
        inputs, labels = torch.tensor(inputs, dtype=torch.float).to(device), torch.tensor(labels, dtype=torch.float).to(device)
        card = len(Y)
        if noise_fn:
            Y = noise_fn(Y)
        model.eval()
        likelihood.eval()
        model.set_train_data(inputs=inputs, targets=labels, strict=False)

        with torch.no_grad():
            query_x = torch.tensor(query_x, dtype=torch.float).to(device)
            pred = likelihood(model(torch.reshape(query_x, [1, config["dim"]])))

        mu = pred.mean.detach().to("cpu").numpy().reshape(-1, )
        stddev = pred.stddev.detach().to("cpu").numpy().reshape(-1, )

        return mu, stddev


parser = argparse.ArgumentParser()
parser.add_argument('--n_iters', default=100, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--space', help='Gym Environment', type=int, default=0)
parser.add_argument('--algorithm', help='RL Algorithm', type=str, choices=["PPO", "DDPG", "A2C", "SAC", "TD3"], default="A2C")
parser.add_argument('--seed', help='Which seed', type=int ,choices=[0,1,2], default=0)
args = parser.parse_args()

spaces =  {"PPO": None,
           "DDPG": None,
           "A2C": None,
           "SAC": None,
           "TD3": None}
with open('config_space_ppo','rb') as f:
    spaces["PPO"] = pickle.load(f)

with open('config_space_ddpg','rb') as f:
    spaces["DDPG"] = pickle.load(f)

with open('config_space_a2c','rb') as f:
    spaces["A2C"] = pickle.load(f)

with open('config_space_ddpg','rb') as f:
    spaces["SAC"] = pickle.load(f)

with open('config_space_ddpg','rb') as f:
    spaces["TD3"] = pickle.load(f)

args.seed = int(args.seed)
print(args.seed)
rootdir = os.path.dirname(os.path.realpath(__file__))
savedir = os.path.join(rootdir, "results", f"seed-{args.seed}", "DyHPO_%s_%s" % (args.algorithm, args.space))
os.makedirs(savedir, exist_ok=True)
config_space = spaces[args.algorithm]

Lambda,response = [], []
c,D = len(config_space[0]), len(config_space[0])
random = np.random.RandomState(args.seed)
randomInitializer = np.random.RandomState(args.seed) ########### for random restarts
log_dir = os.path.join(rootdir, "logs", f"seed-{args.seed}", "DyHPO_%s_%s" % (args.algorithm, args.space))
os.makedirs(log_dir, exist_ok=True)
logger = os.path.join(log_dir, f"{args.algorithm}.txt")
backbone_params = json.load(open(os.path.join(rootdir,"Setconfig90.json"),"rb"))
backbone_params.update({"dim":D})
load_model = False
checkpoint_path = None

random = np.random.RandomState(seed=int(args.seed))
# np.random.seed(args.seed)
# if args.seed == 0:
#     INIT_ID = 12
# elif args.seed == 1:
#     INIT_ID = 23
# elif args.seed == 2:
#     INIT_ID = 9
# n_init = 1
# if args.algorithm == "A2C":
#     x = [(-4, 0.8)]
# elif args.algorithm == "PPO":
#     x = [(-4, 0.8, 0.2)]
# else:
#     x = [(-4, 0.8, 0.0001)]# 0.2)] # [config_space[INIT_ID]]
# x.append(config_space[29])
if args.algorithm in ["PPO", "A2C", "DDPG", "SAC"]:
    with open("init_config_%s_%s_%s" % (args.algorithm, ENVS[int(args.space)], args.seed), 'rb') as f:
        x = pickle.load(f)
else:
    with open("init_config_DDPG_%s_%s" % (ENVS[int(args.space)], args.seed), 'rb') as f:
        x = pickle.load(f)
y = {}
print(x)
init_budget = 100
incumbent = -np.inf
start_time = time.time()
for cfg in x:
    config = None
    if args.algorithm == "PPO":
        config = {
            "lr": cfg[0],
            "gamma": cfg[1],
            "clip": cfg[2]
        }
    elif args.algorithm in ["DDPG", "SAC", "TD3"]:
        config = {
            "lr": cfg[0],
            "gamma": cfg[1],
            "tau": cfg[2]
        }
    elif args.algorithm == "A2C":
        config = {
            "lr": cfg[0],
            "gamma": cfg[1]
        }
    results = get_metrics(search_space=args.algorithm, environment=ENVS[int(args.space)], config=config,
                          seed=args.seed)
    y[cfg] = results["eval_avg_returns"]
    if results["eval_avg_returns"][-1] > incumbent:
        incumbent = results["eval_avg_returns"][-1]
        inc_x = cfg
        inc_budget = 100

    budget = init_budget
    run_data = {"incumbent": results["eval_avg_returns"][-1], "configuraton": cfg, "budget": init_budget,
                "wallclock_time_search": time.time() - start_time,
                "wallclock_time_eval": results["eval_timestamps"][inc_budget-1], "eval_timesteps": results["eval_timesteps"][inc_budget-1],
                "full_wallclock_time_eval": results["eval_timestamps"][-1], "full_eval_timesteps": results["eval_timesteps"][-1]}
    with open("GP_%s_%s_%s_init_config.json" % (args.seed, ENVS[int(args.space)], args.algorithm), "a+") as f:
        json.dump(run_data, f)
        f.write('\n')

random_seed = randomInitializer.randint(0, 100000)
all_y = {}
# for i in range(len(config_space)):
#     cfg = config_space[i]
#     if args.algorithm == "PPO":
#         config = {
#             "lr": cfg[0],
#             "gamma": cfg[1],
#             "clip": cfg[2]
#         }
#     elif args.algorithm == "DDPG":
#         config = {
#             "lr": cfg[0],
#             "gamma": cfg[1],
#             "tau": cfg[2]
#         }
#     elif args.algorithm == "A2C":
#         config = {
#             "lr": cfg[0],
#             "gamma": cfg[1]
#         }
#     results = get_metrics(search_space=args.algorithm, environment=ENVS[int(args.space)], config=config,
#                          seed=args.seed)
#     all_y[config_space[i]] = results["eval_avg_returns"][:init_budget]
# losses, weights, initial_weights = model.train(config_space, all_y, checkpoint=checkpoint_path,
#                                                            epochs=backbone_params["epochs"], optimizer=optimizer,
#                                                            optimizer_nn=optimizer_nn, verbose=True)
retries = 0
for _ in range(args.n_iters):
    done = False
    while not done:
        # try:
        #     model = LCGP(log_dir=logger, kernel=backbone_params["kernel"], support=x, backbone_fn=backbone_fn,
        #                   config=backbone_params, seed=random_seed)

            # optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': backbone_params["lr"]},
            #                       {'params': model.feature_extractor.parameters(), 'lr': backbone_params["lr"]}])
            model, likelihood, mll = get_model_likelihood_mll(1000, len(config_space[0]), backbone_params)
            optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': backbone_params["lr"]}])
            model, likelihood, mll, max_Y = train(model, likelihood, mll, x, y, optimizer, config=backbone_params)

            candidate = propose_location(incumbent=incumbent, X_sample=x, Y_sample=y, gpr=model, likelihood=likelihood,
                                         config_space=config_space,
                                        budget=budget, dim=D, inc_x=inc_x, max_Y=max_Y, conf=backbone_params)
            print(candidate)
            lr = config_space[candidate][0]
            gamma = config_space[candidate][1]
            if args.algorithm in ["PPO", "DDPG", "SAC", "TD3"]:
                clip = config_space[candidate][2]
                cfg_triple = (lr, gamma, clip)
            else:
                cfg_triple = (lr, gamma)
            # clip = config_space[candidate][2]
            if cfg_triple in x:
                # run_budget = min(100, len(y[(lr, gamma)]) + 5)
                print("Selected double!!!!!")
            else:
                run_budget = 100
                x.append(cfg_triple)
            budget = 100
            config = None
            if args.algorithm == "PPO":
                config = {
                    "lr": cfg_triple[0],
                    "gamma": cfg_triple[1],
                    "clip": cfg_triple[2]
                }
            elif args.algorithm in ["DDPG", "SAC", "TD3"]:
                config = {
                    "lr": cfg_triple[0],
                    "gamma": cfg_triple[1],
                    "tau": cfg_triple[2]
                }
            elif args.algorithm == "A2C":
                config = {
                    "lr": cfg_triple[0],
                    "gamma": cfg_triple[1]
                }
            results = get_metrics(search_space=args.algorithm, environment=ENVS[int(args.space)], config=config,
                                  seed=args.seed)
            y[cfg_triple] = results["eval_avg_returns"][:budget]
            print("Queried: %s, %s: %s" % (lr,gamma, results["eval_avg_returns"][-1]))
            # incumbent, inc_x, inc_budget = find_incumbent(y=y, budget=budget)
            if results["eval_avg_returns"][-1] > incumbent:
                incumbent = results["eval_avg_returns"][-1]
                inc_x = cfg_triple
                inc_budget = 100
            print("Budget: %s" % budget)
            dome = True
            # budget += 5
            print("Max budget: %s" % budget)
            print("Incumbent at %s" % (time.time() - start_time))
            print("Config: %s    Budget: %s    Reward: %s" % (inc_x, inc_budget, incumbent))
            run_data = {"incumbent": results["eval_avg_returns"][budget -1], "configuraton": list(cfg_triple), "budget": int(inc_budget),
                        "wallclock_time": float(time.time() - start_time),
                       "wallclock_time_eval": results["eval_timestamps"][budget-1], "eval_timesteps": results["eval_timesteps"][budget-1],
                       "full_wallclock_time_eval": results["eval_timestamps"][-1], "full_eval_timesteps": results["eval_timesteps"][-1]}
            with open("GP_%s_%s_%s_init_config.json" % (args.seed, ENVS[int(args.space)], args.algorithm), "a+") as f:
                json.dump(run_data, f)
                f.write('\n')
            # budget = min(100, budget + 5)
            done = True
        # except Exception as e:
        #     random_seed = randomInitializer.randint(0, 100000)
        #     retries += 1
        #     print(f"retry no. {retries}");
        #     print(e)
print("Max budget: %s" % budget)
print("Incumbent")
print("Config: %s    Budget: %s    Reward: %s" % (inc_x, inc_budget, incumbent))
