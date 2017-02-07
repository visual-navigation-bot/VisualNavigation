# a subclass of Action_Space for discrete action space

from action_space import Action_Space

class Discrete_Action_Space(Action_Space):
    def __init__(self, n_act):
        '''
        Argument:
            n_act: an integer specifying total number of possible actions
        '''
        assert type(n_act)==int, 'n_act in class Discrete_Action_Space must be an integer'
        assert n_act>0, 'n_act in class Discrete_Action_Space must > 0'
        Action_Space.__init__(self)
        self._n_act = n_act
        self._range = range(n_act) 

    @property
    def range(self):
        return self._range

    def __str__(self):
        description = 'Discrete({}), possible actions ='.format(self._n_act)
        for i in range(self._n_act-1):
            description += ' {},'.format(self._range[i])
        description += ' {}'.format(self._range[self._n_act-1])

        return description

def test():
    obj = Discrete_Action_Space(5)
    print(obj)
    print('Discrete_Action_Space.range = {}'.format(obj.range))

if __name__=='__main__':
    test()
