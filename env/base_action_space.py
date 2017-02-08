# base class of all action space 

class Action_Space(object):
    def __init__(self):
        pass

    def sample(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

