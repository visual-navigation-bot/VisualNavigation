# base class of all simulation method

class Simulation(object):
    def __init__(self):
        pass

    def add_ped(self):
        raise NotImplementedError

    def move(self):
        raise NotImplementedError

    def get_ped_state(self):
        '''

        Return:
            a dictionay with keys,
                'ped_ID': pedestrian ID, a (n,) int8 nparray
                'ped_position': pedestrian position, a (n,2) float32 nparray
                'ped_velocity': pedestrian velocity(optional), a (n,2) float32 nparray
        '''
        raise NotImplementedError

    def add_agent(self):
        raise NotImplementedError

