import numpy as np

class OU_Noise(object):
    def __init__(self, act_dim, mu=0, theta=0.15, sigma=0.2):
        self._act_dim = act_dim
        self._mu = mu
        self._theta = theta
        self._sigma = sigma
        self._state = np.ones(self._act_dim) * self._mu

    def reset(self):
        self.state = np.ones(self._act_dim) * self._mu

    def noise(self):
        x = self._state
        dx = self._theta*(self._mu-x) + self._sigma*np.random.randn(len(x))
        self._state = x + dx
        return self._state
