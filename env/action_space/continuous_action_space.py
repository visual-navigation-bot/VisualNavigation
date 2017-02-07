# a subclass of Action_Space for continuous action space

from base_action_space import Action_Space
import numpy as np

class Continuous_Action_Space(Action_Space):
    def __init__(self, low, high):
        '''
        Argument:
            low: a float 1D numpy array, 
            high: a float 1D numpy array
        '''
        assert type(low)==np.ndarray, 'low should be of type numpy.ndarray'
        assert type(high)==np.ndarray, 'high shold be of type numpy.ndarray'
        assert len(low)==len(high), 'low and high should have the same length'

        Action_Space.__init__(self)
        self._low = low
        self._high = high
        self._dim = len(low)

    @property
    def low(self):
        return self._low
    
    @property
    def high(self):
        return self._high

    def __str__(self):
        des = 'Continuous(low,high):'
        for d in range(self._dim):
            des += '\n  {}\'th dimension = ({},{})'.format(d, self._low[d],self._high[d])
        
        return des

