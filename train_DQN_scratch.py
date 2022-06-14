from Agents.agent_1 import AgentPolicy
from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default
from DQN_scratch.DQN_models import DQN, DDQN, PrioritizedReplayDDQN


def train():
    # Run a training job
    configs = LuxMatchConfigs_Default
    configs["width"] = 12
    configs["height"] = 12
    opponent = Agent()

    # DQN
    for i in range(2):
        player = AgentPolicy(mode='train')

        env = LuxEnvironment(configs=configs,
                             learning_agent=player,
                             opponent_agent=opponent)

        model = DQN(env)

        model.train(
            f'./models/DQN_scratch_agent_1/{i}')
        opponent = AgentPolicy(mode='inference', model=model)

    # DDQN
    opponent = Agent()
    for i in range(2):
        player = AgentPolicy(mode='train')

        env = LuxEnvironment(configs=configs,
                             learning_agent=player,
                             opponent_agent=opponent)

        model = DDQN(env)

        model.train(
            f'./models/DDQN_agent_1/{i}')
        opponent = AgentPolicy(mode='inference', model=model)

    # PrioritizedReplayDDQN
    opponent = Agent()
    for i in range(2):
        player = AgentPolicy(mode='train')

        env = LuxEnvironment(configs=configs,
                             learning_agent=player,
                             opponent_agent=opponent)

        model = PrioritizedReplayDDQN(env)

        model.train(
            f'./models/PRDQN_agent_1/{i}')
        opponent = AgentPolicy(mode='inference', model=model)


if __name__ == "__main__":
    train()
