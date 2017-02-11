import sys
sys.path.append('../env')
sys.path.append('../env/env_no_ped_v0')

import numpy as np

from environment import No_Ped_v0

def make():
    step_time = 0.8
    field_size = (100,100)
    nav_path = '~/Desktop/VisualNavigation_local/env/env_no_ped_v0/nav_data/no_ped_v0_1.pkl'
    start_point = np.array([80,20])
    dest_point = np.array([20,80])

    env = No_Ped_v0(step_time, field_size, nav_path)
    params = {
        'start': start_point,
        'destination': dest_point
    }
    env.set_params(params)

    return env
