# observation space for No_Ped_v0

from base_observation_space import Observation_Space

class Obs_Space_No_Ped_v0(Observation_Space):
    def __init__(self, field_size):
        self._field_size = field_size

    def field_size(self):
        return self._field_size

    def __str__(self):
        des = 'Obs_Space_No_Man_v0.__str__'

        return des
