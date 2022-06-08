import torch.nn as nn
import torch.nn.functional as F

import random
from collections import deque

# Credit: HW4 sample code


class replay_buffer():
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
