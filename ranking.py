import os
import importlib
import sys
import io

from functools import cmp_to_key
from stable_baselines3 import PPO, DQN
from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default


class Model:
    def __init__(self, name, agent):
        self.name = name
        self.agent = agent
        self.win_matches = 0

    def increase_winning(self):
        self.win_matches += 1


def rank(directories, matches):
    models = []
    for agent_dir, model_dir, algo in directories:
        for f in os.listdir(model_dir):
            if f.endswith('.zip'):
                path = os.path.join(model_dir, f)
                if algo == 'DQN':
                    model = DQN.load(path)
                elif algo == 'PPO':
                    model = PPO.load(path)
                else:
                    raise ValueError(f'Algorithm {algo} not found.')

                agent_policy_path = agent_dir + '.agent_policy'
                agent_policy = importlib.import_module(agent_policy_path)
                agent = agent_policy.AgentPolicy(mode='inference', model=model)
                models.append(Model(path, agent))

    n = len(models)

    def battle(agent_i, agent_j):
        configs = LuxMatchConfigs_Default
        env = LuxEnvironment(configs=configs,
                             learning_agent=agent_i,
                             opponent_agent=agent_j)
        env.run_no_learn()
        return env.game.get_winning_team()

    # run several matches between i and j, output the stronger side
    # >0: i, =0: tie, <0: j
    def match(model_i, model_j):
        win_i, win_j = 0, 0
        agent_i, agent_j = model_i.agent, model_j.agent
        for i in range(matches):
            result = battle(agent_i, agent_j)
            if result == 0:
                win_i += 1
            else:
                win_j += 1
        return win_i - win_j

    models.sort(key=cmp_to_key(match), reverse=True)
    for i in range(n):
        print('%d %s' % (i + 1, models[i].name))


if __name__ == '__main__':
    with open('agents.txt') as f:
        directories = map(lambda line: tuple(line.split()), f.readlines())
        rank(directories, 11)
