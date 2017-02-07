# a subclass of Action_Space for continuous action space

from action_space import Action_Space

class Continuous_Action_Space(Action_Space):
    def __init__(self, act_range):
        '''
        Argument:
            - act_range: a float list with 2 elements (low, high)
        '''
        assert type(act_range)==list, 'act_range in class Continuous_Action_Space must be a list'
        assert len(act_range)==2, 'act_range in class Continuous_Action_Space must have length as 2'
        Action_Space.__init__(self)
        self._range = act_range

    @property
    def range(self):
        return self._range
   
    def __str__(self):
        description = 'Continuous({}(low), {}(high))'.format(self._range[0],self._range[1])
        return description

def test():
    obj = Continuous_Action_Space([-1.,1.])
    print(obj)
    print('Continuous_Action_Space.range = {}'.format(obj.range))

if __name__=='__main__':
    test()

