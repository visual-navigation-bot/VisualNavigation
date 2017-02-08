import numpy as np
from base_observation_space import Observation_Space

class Observation_Space_LTA_Continuous_ver0(Observation_Space):
    """
    observation: dictionary;
        agent_ID: -1
        agent_position: np.1darray; the position of agent
        agent_velocity: np.1darray; the velocity of agent
        agent_goal_position: np.1darray; the goal position of agent
        ped_ID: np.1darray int8; ID of other pedestrians
        ped_position: np.2darray float32; axis 0 is agent index, axis 1 is agent position
        ped_velocity: np.2darray float32; axis 0 is agent index, axis 1 is agent velocity
    """
    def __init__(self):
        Observation_Space.__init__(self)

    def __str__(self):
        return "a dictionary, please check source code for further explanation"
