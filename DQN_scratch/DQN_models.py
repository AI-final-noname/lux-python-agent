import torch
import torch.nn as nn
import numpy as np
from luxai2021.env.lux_env import LuxEnvironment
from .utilities import ReplayBuffer, Net, PriortizedMemory
from tqdm import tqdm


class DQN():
    def __init__(self, env, epsilon=0.05, learning_rate=0.0001, GAMMA=0.99, batch_size=32, capacity=10000, n_actions=9):
        """
        Hyperparameters:
            epsilon: Determines the explore/expliot rate of the agent
            learning_rate: Determines the step size while moving toward a minimum of a loss function
            GAMMA: the discount factor (tradeoff between immediate rewards and future rewards)
            batch_size: the number of samples which will be propagated through the neural network
            capacity: the size of the replay buffer/memory
        """
        self.env = env

        self.count = 0  # recording the number of iterations

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = GAMMA
        self.capacity = capacity
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.buffer = ReplayBuffer(self.capacity)

        if env == None:
            return
        # the evaluate network
        self.evaluate_net = Net(
            self.env.learning_agent.observation_shape[0], self.n_actions)
        # the target network
        self.target_net = Net(
            self.env.learning_agent.observation_shape[0], self.n_actions)

        # Adam is a method using to optimize the neural network
        self.optimizer = torch.optim.Adam(
            self.evaluate_net.parameters(), lr=self.learning_rate)

    def choose_action(self, state):
        with torch.no_grad():
            if np.random.random() <= 1 - self.epsilon:
                Q = self.target_net.forward(
                    torch.FloatTensor(state)).squeeze(0).detach()
                action = int(torch.argmax(Q).numpy())
            else:
                action = np.random.randint(self.env.action_space.n)
        return action

    def learn(self):
        if self.count % 100 == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())

        # Compute the mask
        states, actions, rewards, next_states, done \
            = self.buffer.sample(self.batch_size)
        non_final_mask = torch.tensor(
            tuple(map(lambda d: not d, done)), dtype=torch.bool)
        non_final_next_states_list \
            = [next_states[i] for i in range(self.batch_size) if non_final_mask[i]]
        non_final_next_states = torch.tensor(
            np.array(non_final_next_states_list), dtype=torch.float)
        states_batch = torch.tensor(
            np.array(states), dtype=torch.float)
        actions_batch = torch.tensor(np.array(actions))
        rewards_batch = torch.tensor(
            np.array(rewards), dtype=torch.float)

        # Get result from evaluation net
        evaluated_values = self.evaluate_net.forward(states_batch) \
            .gather(1, torch.unsqueeze(actions_batch, 1)) \
            .squeeze(1)

        # Get result from target net
        next_states_values = torch.zeros(self.batch_size)
        next_states_values[non_final_mask] \
            = self.target_net.forward(non_final_next_states) \
            .max(1)[0] \
            .detach()
        target_values = self.gamma * next_states_values + rewards_batch

        # Compute the loss
        criterion = nn.MSELoss()
        loss = criterion(target_values, evaluated_values)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, file_prefix, step_count=300000, save_freq=100000):  # main train function
        state = self.env.reset()
        for _ in tqdm(range(step_count)):
            self.count += 1
            action = self.choose_action(state)
            obs, reward, done, _ = self.env.step(action)
            self.buffer.insert(state, int(action), reward, obs, int(done))
            if self.count >= 1000:
                self.learn()
            if done:
                self.env.reset()
            if self.count % save_freq == 0:
                self.save(file_prefix + f'_{self.count}.qt')

    def predict(self, obs, deterministic=False):  # agent.py
        with torch.no_grad():
            Q = self.target_net.forward(
                torch.FloatTensor(obs)).squeeze(0).detach()
            action = int(torch.argmax(Q).numpy())
        return action, None

    def save(self, filename):  # lux_env.py
        torch.save(self.target_net.state_dict(), filename)

    @classmethod
    def load(cls, filename):  # ranking
        ret = cls(None)
        ret.target_net = Net(85, ret.n_actions)
        ret.target_net.load_state_dict(torch.load(filename))
        return ret


class DDQN(DQN):
    def learn(self):
        if self.count % 100 == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())

        # Compute the mask
        states, actions, rewards, next_states, done \
            = self.buffer.sample(self.batch_size)
        non_final_mask = torch.tensor(
            tuple(map(lambda d: not d, done)), dtype=torch.bool)
        non_final_next_states_list \
            = [next_states[i] for i in range(self.batch_size) if non_final_mask[i]]
        non_final_next_states = torch.tensor(
            np.array(non_final_next_states_list), dtype=torch.float)
        states_batch = torch.tensor(
            np.array(states), dtype=torch.float)
        actions_batch = torch.tensor(np.array(actions))
        rewards_batch = torch.tensor(
            np.array(rewards), dtype=torch.float)

        # Get result from evaluation net
        evaluated_values = self.evaluate_net.forward(states_batch) \
            .gather(1, torch.unsqueeze(actions_batch, 1)) \
            .squeeze(1)

        # Get result from target net
        next_states_values = torch.zeros(self.batch_size)
        next_states_values_actions = self.evaluate_net.forward(
            non_final_next_states).argmax(1)
        next_states_values[non_final_mask] \
            = self.target_net.forward(non_final_next_states) \
            .gather(1, torch.unsqueeze(next_states_values_actions, 1)) \
            .squeeze(1)
        target_values = self.gamma * next_states_values + rewards_batch

        # Compute the loss
        criterion = nn.MSELoss()
        loss = criterion(target_values, evaluated_values)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class PrioritizedReplayDDQN(DQN):
    def __init__(self, env, epsilon=0.05, learning_rate=0.0001, GAMMA=0.99, batch_size=32, capacity=10000):
        super().__init__(env, epsilon, learning_rate,
                         GAMMA, batch_size, capacity)
        self.buffer = PriortizedMemory(self.capacity)

    def learn(self):
        if self.count % 100 == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())

        # Compute the mask
        batch, idxs, is_weights \
            = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, done = zip(*batch)
        non_final_mask = torch.tensor(
            tuple(map(lambda d: not d, done)), dtype=torch.bool)
        non_final_next_states_list \
            = [next_states[i] for i in range(self.batch_size) if non_final_mask[i]]
        non_final_next_states = torch.tensor(
            np.array(non_final_next_states_list), dtype=torch.float)
        states_batch = torch.tensor(
            np.array(states), dtype=torch.float)
        actions_batch = torch.tensor(np.array(actions))
        rewards_batch = torch.tensor(
            np.array(rewards), dtype=torch.float)

        # Get result from evaluation net
        evaluated_values = self.evaluate_net.forward(states_batch) \
            .gather(1, torch.unsqueeze(actions_batch, 1)) \
            .squeeze(1)

        # Get result from target net
        next_states_values = torch.zeros(self.batch_size)
        next_states_values_actions = self.evaluate_net.forward(
            non_final_next_states).argmax(1)
        next_states_values[non_final_mask] \
            = self.target_net.forward(non_final_next_states) \
            .gather(1, torch.unsqueeze(next_states_values_actions, 1)) \
            .squeeze(1)
        target_values = self.gamma * next_states_values + rewards_batch

        # Update priority
        errors = torch.abs(evaluated_values - target_values).data.numpy()
        for i in range(self.batch_size):
            self.buffer.update(idxs[i], errors[i])

        # Compute the loss
        criterion = nn.MSELoss()
        loss = (torch.FloatTensor(is_weights) *
                criterion(target_values, evaluated_values)).mean()
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
