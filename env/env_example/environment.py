import numpy as np

from action_space.continuous_action_space import Continuous_Action_Space
from observation_space import Obs_Space_Example
from base_environment import Environment

class Example(Environment):
    def __init__(self, step_time, field_size, extra):
        act_low = np.array([0.])
        act_high = np.array([1.])

        # need to be done for every Environment subclass
        Environment.__init__(self, step_time, field_size)
        self._step_time = step_time
        self._field_size = field_size
        self._action_space = Continuous_Action_Space(act_low,act_high)
        self._observation_space = Obs_Space_Example(0., 4.)

        # extra argument for different subclass
        self._extra = extra #unused
        
        # non-argument-assigned variable
        self._x = None
        self._r = None

    def step(self, action):
        print('Example.step(action) called')
        # make sure action follows the rule of self._action_space
        self._x += action
        
        observation = self._x
        if self._x<=self._observation_space.high:
            self._r += 1
            done = False
        else:
            self._r = 0
            done = True
        reward = self._r

        return observation, reward, done

    def display(self):
        print('Example.display() called, x = {}'.format(self._x))
        # should start pygame display

    def reset(self):
        print('Example.reset() called')
        self._x = self._observation_space.low
        self._r = 0

        observation = self._x

        return observation

    def set_params(self, params):
        print('Example.set_params(params) called')
        self._x = params['x']

    def __str__(self):
        des = 'This environment maintain an internal variable x.\n'
        des += 'At each step, it will add x by action and if x is\n'
        des += 'smaller than observation_space.low, reward += 1.\n'
        des += 'Otherwise, reward = 0 and terminate'

        return des

