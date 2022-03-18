from enum import Enum

import numpy as np


class ReplayMemory:
    class Buffers(Enum):
        STATE = 0
        ACTION = 1
        REWARD = 2
        NEXT_STATE = 3
        DONE = 4

    def __init__(self):
        self.max_length = 10000
        self.length = 0
        self.replay_memory = [[] for _ in range(len(ReplayMemory.Buffers))]

    def sample(self, batch_size):
        sample_indices = np.random.choice(range(self.length), batch_size)
        sample_buffers = []
        for buffer in self.replay_memory:
            sample_buffers.append(np.array(buffer).take(sample_indices, axis=0))
        return sample_buffers

    def register(self, entry):
        assert len(entry) == len(ReplayMemory.Buffers)
        for i, buffer in enumerate(self.replay_memory):
            buffer.append(entry[i])
        self.length += 1
        if self.length > self.max_length:
            self.length -= 1
            for buffer in self.replay_memory:
                del buffer[:1]

    def reset(self):
        self.length = 0
        del self.replay_memory
        self.replay_memory = [[] for _ in range(len(ReplayMemory.Buffers))]


class Agent:
    def __init__(self, state_shape, action_num):
        self.state_shape = state_shape
        self.action_num = action_num
        self.epsilon = 1.0
        self.max_processed_steps = 1000000
        self.replay_memory = ReplayMemory()

    def explore(self):
        if self.epsilon > 0.01:
            self.epsilon *= 0.99
        if np.random.random() < self.epsilon:
            return np.random.choice(range(self.action_num))
        return None

    def move(self, state):
        pass

    def train(self, env):
        pass