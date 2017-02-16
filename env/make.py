# environment
from env_example.environment import Example
import env_LTA_Continuous_ver0.example
import env_LTA_Discrete_ver0.example

# navigation map
from nav_v0.navigation_map import Navigation_Map_v0

def make_environment(env_name, step_time=0.4, field_size=(800,600), params = {}):
    if env_name=='Example':
        x_start = 50
        return Example(step_time, field_size, x_start)
    elif env_name=='LTA_Continuous_ver0_Opposite_Walk':
        return env_LTA_Continuous_ver0.example.Opposite_Walk(
               step_time, field_size, params)
    elif env_name=='LTA_Discrete_ver0_Opposite_Walk':
        return env_LTA_Discrete_ver0.example.Opposite_Walk(
               step_time, field_size, params)
    elif env_name=='LTA_Continuous_ver0_Two_Peds_Walk':
        return env_LTA_Continuous_ver0.example.Two_Peds_Walk(
                step_time, field_size, params)
    else:
        raise NotImplementedError('No such enviroment {} implmented'.format(env_name))

def make_navigation_map(nav_name):
    if nav_name=='Navigation_Map_v0':
        print('do nothing, undone')#TODO 
    else:
        raise NotImplementedError('No such navigation map {} implemented'.format(nav_name))


