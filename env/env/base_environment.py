# base class of all environments

class Environment(object):
    def __init__(self, step_time, field_size):
        '''
        FUNC: constructor of Environment class
        Argument:
            step_time: a float, time interval of taking steps
            field_size: an interger tuple with 2 elements (width, height)
        '''
        self._step_time = step_time
        self._action_space = None # an object
        self._observation_space = None # an object

    def step(self, action):
        '''
        FUNC: agent take an action and interact with the environment
        Argument:
            - action
        Return:
            - observation
            - reward
            - done
        '''
        raise NotImplementedError

    def display(self):
        '''
        FUNC: alias to 'render()' in gym
        '''
        raise NotImplementedError

    def reset(self):
        '''
        FUNC: reset environment
        Return:
            - observation
        '''
        raise NotImplementedError

    def set_params(self, params):
        '''
        Argument:
            params: a dictionary containing parameters to be modified
        '''
        raise NotImplementedError

    @property
    def action_space(self):
        if self._action_space is not None:
            return self._action_space
        else:
            raise ValueError('Action space is undefined.')
    @property
    def observation_space(self):
        if self._observation_space is not None:
            return self._observation_space
        else:
            raise ValueError('Observation space is undefined.')

    def __str__(self):
        raise NotImplementedError

