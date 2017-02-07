# parent class of all environments

class Environment(object):
    def __init__(self, step_time):
        '''
        FUNC: constructor of Environment class
        Argument:
            - step_time: a float, time interval of taking steps
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

def test():
    from discrete_action_space import Discrete_Action_Space
    from continuous_action_space import Continuous_Action_Space
    class Test_Env(Environment):
        def __init__(self, step_time):
            Environment.__init__(self, step_time)
            #self._action_space = Discrete_Action_Space(3)
            self._action_space = Continuous_Action_Space([-1.,1.])
        
        def __str__(self):
            return 'I am an instance of Test_Env'

    obj = Test_Env(0.2)
    act_space = obj.action_space
    print(obj)
    print('Test_Env.action_space {}'.format(act_space))

if __name__=='__main__':
    test()
