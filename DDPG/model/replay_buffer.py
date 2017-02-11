import collections
import random
import numpy as np
import time

class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self._buffer_size = buffer_size
        self._count = 0
        self._buffer = collections.deque()
        # random seed
        random.seed(time.time())

    def add(self, o, a, r, t, o2):
        # add a sample of experience into replay buffer
        # experience x = (observation_t, action_t, reward_t, terminal, observation_(t+1))
        x = tuple((o,a,r,t,o2))
        if self._count<self._buffer_size:
            self._buffer.append(x)
            self._count += 1
        else:
            # FIFO
            self._buffer.popleft()
            self._buffer.append(x)

    def sample_batch(self, batch_size):
        # sample a batch of experience from replay buffer
        return random.sample(self._buffer, batch_size)

    def clear(self):
        # clear the replay buffer
        self._buffer.clear()
        self._count = 0

    @property
    def size(self):
        return self._count
