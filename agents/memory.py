import random
import numpy as np
import torch
from functools import reduce


class DQNReplayMemory:
    """

    """
    def __init__(self, maxlen):
        """

        :param maxlen:
        """
        self.maxlen = maxlen
        self.memory = []
        self.position = 0

    def append(self, experience):
        """

        :param experience:
        :return:
        """
        if len(self.memory) < self.maxlen:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.maxlen

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNFSReplayMemory:
    """

    """
    def __init__(self, maxlen):
        """

        :param maxlen:
        """
        self.maxlen = maxlen
        self.memory = []
        self.position = 0
        self.unroll_steps = 4

    def append(self, experience):
        """

        :param experience:
        :return:
        """
        if len(self.memory) < self.maxlen:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.maxlen

    def sample(self, batch_size):
        """

        :param batch_size:
        :return:
        """
        indices = np.random.randint(low=self.unroll_steps - 1, high=len(self.memory) - 1, size=batch_size)
        indices = [self.consecutive_frame_id(idx) for idx in indices]
        samples = [reduce(lambda t0, t1:
                          (torch.cat((t0[0], t1[0]), dim=1),  # concatenates consecutive frames -> states
                           t1[1],  # takes element of next / last experience tuple -> action of last frame
                           t1[2],  # takes element of next / last experience tuple -> reward of last frame
                           torch.cat((t0[3], t1[3]), dim=1),  # concatenates consecutive frames -> next_states
                           t1[4]),  # takes element of next / last experience tuple -> done of last frame
                          [self.memory[idx - i] for i in range(self.unroll_steps - 1, -1, -1)])
                   for idx in indices]
        return samples

    def consecutive_frame_id(self, idx):
        """

        :param idx:
        :return:
        """
        for i in range(self.unroll_steps):
            if self.memory[idx - i][4]:
                # done == True, return new index
                return idx - i
        # done == False
        return idx

    def __len__(self):
        """

        :return:
        """
        return len(self.memory)


class DARQNReplayMemory:
    """

    """
    def __init__(self, maxlen):
        """

        :param maxlen:
        """
        self.maxlen = maxlen
        self.memory = []
        self.position = 0
        self.unroll_steps = 4

    def append(self, experience):
        """

        :param experience:
        :return:
        """
        if len(self.memory) < self.maxlen:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.maxlen

    def sample(self, batch_size):
        """

        :param batch_size:
        :return:
        """
        indices = np.random.randint(low=self.unroll_steps - 1, high=len(self.memory) - 1, size=batch_size)
        indices = [self.consecutive_frame_id(idx) for idx in indices]
        samples = [reduce(lambda t0, t1:
                          (torch.cat((t0[0], t1[0]), dim=0),  # concatenates consecutive frames -> states
                           t1[1],  # takes element of next / last experience tuple -> action of last frame
                           t1[2],  # takes element of next / last experience tuple -> reward of last frame
                           torch.cat((t0[3], t1[3]), dim=0),  # concatenates consecutive frames -> next_states
                           t1[4]),  # takes element of next / last experience tuple -> done of last frame
                          [self.memory[idx - i] for i in range(self.unroll_steps - 1, -1, -1)])
                   for idx in indices]
        return samples

    def consecutive_frame_id(self, idx):
        """

        :param idx:
        :return:
        """
        for i in range(self.unroll_steps):
            if self.memory[idx - i][4]:
                # done == True, return new index
                return idx - i
        # done == False
        return idx

    def __len__(self):
        """

        :return:
        """
        return len(self.memory)
