import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np
from collections import deque

'''
The following codes are modified from HW4 sample code and the following two implementations:
https://github.com/rlcode/per
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
'''


class ReplayBuffer():
    '''
    A deque storing trajectories
    '''

    def __init__(self, capacity):
        self.capacity = capacity  # the size of the replay buffer
        self.memory = deque(maxlen=capacity)  # replay buffer itself

    def insert(self, state, action, reward, next_state, done):
        '''
        Insert a sequence of data gotten by the agent into the replay buffer.

        Parameter:
            state: the current state
            action: the action done by the agent
            reward: the reward agent got
            next_state: the next state
            done: the status showing whether the episode finish

        Return:
            None
        '''
        self.memory.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        '''
        Sample a batch size of data from the replay buffer.

        Parameter:
            batch_size: the number of samples which will be propagated through the neural network

        Returns:
            observations: a batch size of states stored in the replay buffer
            actions: a batch size of actions stored in the replay buffer
            rewards: a batch size of rewards stored in the replay buffer
            next_observations: a batch size of "next_state"s stored in the replay buffer
            done: a batch size of done stored in the replay buffer
        '''
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, done = zip(*batch)
        return observations, actions, rewards, next_observations, done


class Net(nn.Module):
    '''
    The structure of the Neural Network calculating Q values of each state.
    '''

    def __init__(self, input_state, output_state, hidden_layer_size=50):
        super(Net, self).__init__()
        self.input_state = input_state  # the dimension of state space
        self.output_state = output_state  # the dimension of action space
        self.fc1 = nn.Linear(input_state, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, output_state)

    def forward(self, states):
        '''
        Forward the state to the neural network.

        Parameter:
            states: a batch size of states

        Return:
            q_values: a batch size of q_values
        '''
        x = F.relu(self.fc1(states))
        q_values = self.fc2(x)
        return q_values


class SumTree:
    '''
    A binary tree data structure, where the sum of parent is the sum of childrens.
    Also calculates the maximum of each node to speed up calculation.
    '''
    class Node:
        def __init__(self, weight=0):
            self.sum = weight
            self.max = weight

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = [self.Node() for i in range(2 * capacity - 1)]
        self.data = [None for i in range(capacity)]
        self.n_entries = 0
        self.pointer = 0

    def _propogate_node(self, idx):
        while idx > 0:
            parent = (idx - 1) // 2
            lnode = parent * 2 + 1
            rnode = parent * 2 + 2

            self.tree[parent].__init__()
            if lnode < len(self.tree):
                self.tree[parent].sum += self.tree[lnode].sum
                self.tree[parent].max \
                    = max(self.tree[parent].max, self.tree[lnode].max)
            if rnode < len(self.tree):
                self.tree[parent].sum += self.tree[rnode].sum
                self.tree[parent].max \
                    = max(self.tree[parent].max, self.tree[rnode].max)
            idx = parent

    def _get_index(self, s):
        current_idx = 0
        while True:
            lnode = current_idx * 2 + 1
            rnode = current_idx * 2 + 2
            if lnode >= len(self.tree):
                return current_idx
            if s <= self.tree[lnode].sum:
                current_idx = lnode
            else:
                s -= self.tree[lnode].sum
                current_idx = rnode

    def update(self, idx, p):
        self.tree[idx].__init__(p)
        self._propogate_node(idx)

    def add(self, p, data):
        idx = self.pointer + self.capacity - 1

        self.data[self.pointer] = data
        self.update(idx, p)

        self.pointer += 1
        if self.pointer >= self.capacity:
            self.pointer = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def get(self, s):
        idx = self._get_index(s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx].sum, self.data[dataIdx])

    @property
    def total_p(self):
        return self.tree[0].sum

    @property
    def max_p(self):
        return self.tree[0].max


class PriortizedMemory:
    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return np.power((np.abs(error) + self.epsilon), self.alpha)

    def insert(self, state, action, reward, next_state, done):
        p = self.tree.max_p
        if p == 0:
            p = self.abs_err_upper
        self.tree.add(p, [state, action, reward, next_state, done])

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total_p / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total_p
        is_weight = np.power(self.tree.n_entries *
                             sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
