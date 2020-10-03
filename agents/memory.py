import random


class ReplayMemory:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.memory = []
        self.position = 0

    def append(self, experience):
        if len(self.memory) < self.maxlen:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.maxlen

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
