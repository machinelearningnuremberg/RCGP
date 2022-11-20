"""
Example 1 - Local and Sequential
================================

"""
import logging
logging.basicConfig(level=logging.WARNING)

import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
from arlbechworker import MyWorker



parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--min_budget',   type=float, help='Minimum budget used during the optimization.',    default=5)
parser.add_argument('--max_budget',   type=float, help='Maximum budget used during the optimization.',    default=100)
parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=200)
parser.add_argument('--seed', type=int,   help='Seed', default=0)
parser.add_argument('--algorithm', type=int,   help='Algorithm', default=0)
parser.add_argument('--environment', type=int,   help='Environment', default=0)

args=parser.parse_args()

ALGORITHMS = ["PPO", "DDPG", "A2C"]
ALGORITHM = ALGORITHMS[int(args.algorithm)]
ENVS = ['Pong-v0', 'Alien-v0', 'BankHeist-v0',
        'BeamRider-v0', 'Breakout-v0', 'Enduro-v0',
        'Phoenix-v0', 'Seaquest-v0', 'SpaceInvaders-v0',
        'Riverraid-v0', 'Tennis-v0', 'Skiing-v0', 'Boxing-v0',
        'Bowling-v0', 'Asteroids-v0',
        'Ant-v2', 'Hopper-v2', 'Humanoid-v2',
        'CartPole-v1', 'MountainCar-v0', 'Acrobot-v1', 'Pendulum-v0']
ENV = ENVS[int(args.environment)]
# Step 1: Start a nameserver
# Every run needs a nameserver. It could be a 'static' server with a
# permanent address, but here it will be started for the local machine with the default port.
# The nameserver manages the concurrent running workers across all possible threads or clusternodes.
# Note the run_id argument. This uniquely identifies a run of any HpBandSter optimizer.
NS = hpns.NameServer(run_id='example%s%s' % (ENV, ALGORITHM), host='127.0.0.1', port=None)
NS.start()

# Step 2: Start a worker
# Now we can instantiate a worker, providing the mandatory information
# Besides the sleep_interval, we need to define the nameserver information and
# the same run_id as above. After that, we can start the worker in the background,
# where it will wait for incoming configurations to evaluate.
w = MyWorker(ALGORITHM=ALGORITHM, ENVIRONMENT=ENV, SEED=args.seed, nameserver='127.0.0.1',run_id='example%s%s' % (ENV, ALGORITHM))
w.run(background=True)

# Step 3: Run an optimizer
# Now we can create an optimizer object and start the run.
# Here, we run BOHB, but that is not essential.
# The run method will return the `Result` that contains all runs performed.
bohb = BOHB(  configspace = w.get_configspace(ALGORITHM=ALGORITHM),
              run_id = 'example%s%s' % (ENV, ALGORITHM), nameserver='127.0.0.1',
              min_budget=args.min_budget, max_budget=args.max_budget
           )
res = bohb.run(n_iterations=args.n_iterations)

# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
bohb.shutdown(shutdown_workers=True)
NS.shutdown()

# Step 5: Analysis
# Each optimizer returns a hpbandster.core.result.Result object.
# It holds informations about the optimization run like the incumbent (=best) configuration.
# For further details about the Result object, see its documentation.
# Here we simply print out the best config and some statistics about the performed runs.
id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()

print('Best found configuration:', id2config[incumbent]['config'])
print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
print('A total of %i runs where executed.' % len(res.get_all_runs()))
print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in res.get_all_runs()])/args.max_budget))