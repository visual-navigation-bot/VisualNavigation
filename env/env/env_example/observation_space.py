# observation space for Example

from base_observation_space import Observation_Space

class Obs_Space_Example(Observation_Space):
    def __init__(self, low, high):
        self._high = high
        self._low = low

    @property
    def high(self):
        return self._high
    @property
    def low(self):
        return self._low

    def __str__(self):
        description = 'observation is a float, ranging from {} to {}'.format(self._low, self._high)
        return description

