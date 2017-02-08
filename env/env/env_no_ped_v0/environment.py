from __future__ import division
from __future__ import print_function

import numpy as np
import math
import os
import cv2

from action_space.continuous_action_space import Continuous_Action_Space
from base_environment import Environment
from observation_space import ???#TODO

PI = math.pi

class No_Ped_v0(Environment):
    def __init__(self, step_time, field_size, nmap_path):
        act_low = np.array([0.,-PI])
        act_high = np.array([2.,PI])

        Environment.__init__(self, step_time, field_size)
        self._step_time = step_time
        self._field_size = field_size
        self._action_space = Continuous_Action_Space(act_low, act_high)

        #TODO
        self._nmap = ???

    def step(self, action):
        

        # return obs, r, t

    def display(self):
        pass

    def reset(self):
        pass
        #return obs

    def set_params(self, params):
        pass

    def __str__(self):
        des = '?'
        
        return des
        

