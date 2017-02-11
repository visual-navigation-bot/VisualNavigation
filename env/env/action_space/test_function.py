import numpy as np

from action_space.discrete_action_space import Discrete_Action_Space
from action_space.continuous_action_space import Continuous_Action_Space

def test_discrete():
    obj = Discrete_Action_Space(5)
    print(obj)
    print('Discrete_Action_Space.range = {}'.format(obj.range))

def test_continuous():
    low = np.array([-1.,-2.])
    high = np.array([1.,2.])
    obj = Continuous_Action_Space(low,high)
    print(obj)
    print('Continuous_Action_Space.low/high = {}/{}'.format(obj.low, obj.high))

