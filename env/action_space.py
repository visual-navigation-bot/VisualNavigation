# parent class of Action_Space 

class Action_Space(object):
    def __init__(self):
        pass

    @property
    def range(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError
