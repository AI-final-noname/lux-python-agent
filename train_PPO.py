import argparse

from stable_baselines3 import PPO  # pip install stable-baselines3
from Agents.agent_1 import AgentPolicy
from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment, SaveReplayAndModelCallback
from luxai2021.game.constants import LuxMatchConfigs_Default


def get_command_line_arguments():
    """
    Get the command line arguments
    :return:(ArgumentParser) The command line arguments as an ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description='Training script for Lux RL agent.')
    parser.add_argument('--learning_rate',
                        help='Learning rate', type=float, default=0.001)
    parser.add_argument('--gamma', help='Gamma', type=float, default=0.995)
    parser.add_argument('--gae_lambda', help='GAE Lambda',
                        type=float, default=0.95)
    parser.add_argument('--batch_size', help='batch_size',
                        type=int, default=2048)  # 64
    parser.add_argument(
        '--step_count', help='Total number of steps to train', type=int, default=300000)
    parser.add_argument(
        '--n_steps', help='Number of experiences to gather before each learning period', type=int, default=2048)
    parser.add_argument(
        '--path', help='Path to a checkpoint to load to resume training', type=str, default=None)
    args = parser.parse_args()

    return args


def train(args):
    """
    The main training loop
    :param args: (ArgumentParser) The command line arguments
    """
    configs = LuxMatchConfigs_Default
    opponent = Agent()
    for i in range(2):
        player = AgentPolicy(mode='train')

        env = LuxEnvironment(configs=configs,
                             learning_agent=player,
                             opponent_agent=opponent)

        model = PPO("MlpPolicy",
                    env,
                    verbose=1,
                    tensorboard_log="./lux_tensorboard/",
                    learning_rate=args.learning_rate,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    batch_size=args.batch_size,
                    n_steps=args.n_steps
                    )

        callbacks = []

        player_replay = AgentPolicy(mode='inference', model=model)
        callbacks.append(SaveReplayAndModelCallback(
            save_freq=100000,
            save_path='./models/PPO_agent_1',
            name_prefix=f'{i}',
            replay_env=LuxEnvironment(
                configs=configs,
                learning_agent=player_replay,
                opponent_agent=opponent
            ), replay_num_episodes=5))

        model.learn(total_timesteps=args.step_count,
                    callback=callbacks)

        opponent = AgentPolicy(mode="inference", model=model)


if __name__ == "__main__":
    # Get the command line arguments
    local_args = get_command_line_arguments()

    # Train the model
    train(local_args)
