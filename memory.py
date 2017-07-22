import random as rand
from collections import deque


class Memory:

    def __init__(self, size, batch_size):
        self.batch_size = batch_size
        self.memory = deque(maxlen=size)

    def __len__(self):
        return len(self.memory)
        
    def add(self, state, action, reward, next_state, terminal, end_life):
        if len(self.memory) >= self.memory.maxlen:
            self.memory.popleft()
        self.memory.append( (state, action, reward, next_state, terminal, end_life) )

    def getSample(self):
        return rand.sample(list(self.memory), self.batch_size)

    def reset(self):
        self.memory.clear()

    def update_q_values(self):
        for i in reversed(range(len(self.memory) - 1)):
            action = self.memory[i][1]
            reward = self.memory[i][2]
            end_episode = self.memory[i][4]

