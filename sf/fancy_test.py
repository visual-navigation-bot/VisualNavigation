import simulation as sim
import pygame
import numpy as np
import random

# This is tests fancy version tests

def test1():
    sim_env = sim.Sim('Pedestrian Simulation')
    """
    adding obstacles
    """
    param = {
             'start': np.array([100,100]), 
             'end': np.array([700,100])
             }
    sim_env.add_object('obs', param)

    """
    adding pedestrians
    """
    def dist_obstacle(s):
        """
        calculate the closest vector distant from the ped to obstacle
        Input:
            s: np.1darray; the location of the pedestrian
        """
        if s[1] - 100 < 500 - s[1]:
            return np.array([0., 100. - s[1]])
        if s[1] - 100 > 500 - s[1]:
            return np.array([0., 500. - s[1]])
        return np.array([100000, 100000])

    param = {
             'init_s': np.array([100., 150.]),
             'exp_s': np.array([900., 150.]),
             'v0': 100,
             'vmax': 130,
             'init_v': np.array([100., 0.]),
             'tau': 0.5,
             'V_obs': 100.,
             'R_obs': 500.,
             'sight_angle': np.pi / 2, 
             'sight_const': 0.5,
             'friend': np.array([]),
             'V_others': np.array([]),
             'R_others': np.array([]),
             'dist_obs_func': dist_obstacle
             }
    sim_env.add_object('ped', param)

    param = {
             'init_s': np.array([100., 200.]),
             'exp_s': np.array([900., 200.]),
             'v0': 100,
             'vmax': 130,
             'init_v': np.array([100., 0.]),
             'tau': 0.5,
             'V_obs': 100.,
             'R_obs': 500.,
             'sight_angle': np.pi / 2, 
             'sight_const': 0.5,
             'friend': np.array([True]),
             'V_others': np.array([100.]),
             'R_others': np.array([50.]),
             'dist_obs_func': dist_obstacle
             }
    sim_env.add_object('ped', param)

    param = {
             'init_s': np.array([100., 250.]),
             'exp_s': np.array([900., 250.]),
             'v0': 100,
             'vmax': 130,
             'init_v': np.array([100., 0.]),
             'tau': 0.5,
             'V_obs': 100.,
             'R_obs': 500.,
             'sight_angle': np.pi / 2, 
             'sight_const': 0.5,
             'friend': np.array([False, True]),
             'V_others': np.array([100., 100.]),
             'R_others': np.array([50., 50.]),
             'dist_obs_func': dist_obstacle
             }
    sim_env.add_object('ped', param)

    sim_env.run()
test()

