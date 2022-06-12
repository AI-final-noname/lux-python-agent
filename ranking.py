import os
import importlib
import sys
import io

from stable_baselines3 import PPO, DQN
import DQN_scratch.DQN_models as scratch
from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default
import csv


class Model:
    def __init__(self, name, agent):
        self.name = name
        self.agent = agent


def rank(directories, filename, matches=100):
    models = []
    for agent_dir, model_dir, algo in directories:
        for f in os.listdir(model_dir):
            if f.endswith('.zip') or f.endswith('.qt'):
                path = os.path.join(model_dir, f)
                if algo == 'DQN':
                    model = DQN.load(path)
                elif algo == 'PPO':
                    model = PPO.load(path)
                elif algo == 'DQN_scratch':
                    model = scratch.DQN.load(path)
                elif algo == 'DDQN':
                    model = scratch.DDQN.load(path)
                elif algo == 'PRDQN':
                    model = scratch.PrioritizedReplayDDQN.load(path)
                else:
                    raise ValueError(f'Algorithm {algo} not found.')

                agent_policy_path = agent_dir + '.agent_policy'
                agent_policy = importlib.import_module(agent_policy_path)
                agent = agent_policy.AgentPolicy(mode='inference', model=model)
                models.append(Model(path, agent))

    n = len(models)

    def battle(agent_i, agent_j):
        configs = LuxMatchConfigs_Default
        configs["width"] = 12
        configs["height"] = 12
        env = LuxEnvironment(configs=configs,
                             learning_agent=agent_i,
                             opponent_agent=agent_j)
        env.run_no_learn()
        return env.game.get_winning_team()

    # run several matches between i and j, output the stronger side
    # >0: i, =0: tie, <0: j
    def match(model_i, model_j):
        points_i, points_j = 0.0, 0.0
        agent_i, agent_j = model_i.agent, model_j.agent
        for i in range(matches):
            result = battle(agent_i, agent_j)
            if result == 0:
                points_i += 1
            elif result == 1:
                points_j += 1
            else:
                points_i += 0.5
                points_j += 0.5
        return points_i, points_j

    win = [[0.0 for _ in range(len(models))] for _ in range(len(models))]
    for i in range(len(models)):
        for j in range(i):
            points_i, points_j = match(models[i], models[j])
            win[i][j] = points_i
            win[j][i] = points_j

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        agent_names = [[agent.name for agent in models]]
        win = agent_names + win
        writer.writerows(win)


if __name__ == '__main__':
    with open('agents.txt') as f:
        directories = map(lambda line: tuple(line.split()), f.readlines())
        for idx, directory in enumerate(directories):
            rank([directory], f'result/kirito_PPO_DQN/{idx}.txt')
