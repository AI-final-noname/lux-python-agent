import argparse
import glob
import os
import sys
import random

from stable_baselines3 import PPO, DQN  # pip install stable-baselines3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed, get_schedule_fn
from stable_baselines3.common.vec_env import SubprocVecEnv

from agent_policy import AgentPolicy
from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment, SaveReplayAndModelCallback
from luxai2021.game.constants import LuxMatchConfigs_Default
from DQN_models import DDQN


# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html?highlight=SubprocVecEnv#multiprocessing-unleashing-the-power-of-vectorized-environments
def make_env(local_env, rank, seed=0):
    """
    Utility function for multi-processed env.

    :param local_env: (LuxEnvironment) the environment
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        local_env.seed(seed + rank)
        return local_env

    set_random_seed(seed)
    return _init


def get_command_line_arguments():
    """
    Get the command line arguments
    :return:(ArgumentParser) The command line arguments as an ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description='Training script for Lux RL agent.')
    parser.add_argument('--id', help='Identifier of this run',
                        type=str, default=str(random.randint(0, 10000)))
    parser.add_argument('--learning_rate',
                        help='Learning rate', type=float, default=0.001)
    parser.add_argument('--gamma', help='Gamma', type=float, default=0.995)
    parser.add_argument('--gae_lambda', help='GAE Lambda',
                        type=float, default=0.95)
    parser.add_argument('--batch_size', help='batch_size',
                        type=int, default=2048)  # 64
    parser.add_argument(
        '--step_count', help='Total number of steps to train', type=int, default=500000)
    parser.add_argument(
        '--n_steps', help='Number of experiences to gather before each learning period', type=int, default=2048)
    parser.add_argument(
        '--path', help='Path to a checkpoint to load to resume training', type=str, default=None)
    parser.add_argument(
        '--n_envs', help='Number of parallel environments to use in training', type=int, default=1)
    args = parser.parse_args()

    return args


def train(args):
    """
    The main training loop
    :param args: (ArgumentParser) The command line arguments
    """
    # Run a training job
    configs = LuxMatchConfigs_Default
    opponent = Agent()
    for i in range(2):
        player = AgentPolicy(mode='train')

        env = LuxEnvironment(configs=configs,
                             learning_agent=player,
                             opponent_agent=opponent)

        model = DDQN(env)

        print('Training model...')
        model.train(step_count=1000)
        print('Done Training model.')

        opponent = AgentPolicy(mode="inference", model=model)

        # env=LuxEnvironment(configs=configs,
        #                  learning_agent=AgentPolicy(
        #                      mode="inference", model=model),
        #                  opponent_agent=opponent)
        #
        # env.run_no_learn()
        # print(env.game.get_winning_team())
    print("Done")


if __name__ == "__main__":
    if sys.version_info < (3, 7) or sys.version_info >= (3, 8):
        os.system("")

        class style():
            YELLOW = '\033[93m'
        version = str(sys.version_info.major) + "." + \
            str(sys.version_info.minor)
        message = f'/!\ Warning, python{version} detected, you will need to use python3.7 to submit to kaggle.'
        message = style.YELLOW + message
        print(message)

    # Get the command line arguments
    local_args = get_command_line_arguments()

    # Train the model
    train(local_args)
