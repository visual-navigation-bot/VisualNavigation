# environment
from env_example.environment import Example

# navigation map
from nav_v0.navigation_map import Navigation_Map_v0

def make_environment(env_name, step_time=0.2, field_size=(400,600)):
    if env_name=='Example':
        x_start = 50
        return Example(step_time, field_size, x_start)
    else:
        raise NotImplementedError('No such enviroment {} implmented'.format(env_name))

def make_navigation_map(nav_name):
    if nav_name=='Navigation_Map_v0':
        print('do nothing, undone')#TODO 
    else:
        raise NotImplementedError('No such navigation map {} implemented'.format(nav_name))
