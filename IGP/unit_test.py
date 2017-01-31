import pygame
import numpy as np
import simulation
import time
import util

x_kernel = util.summed_kernel(
        util.matern_kernel(np.exp(3.3434), np.exp(2*4.5640)),
        util.linear_kernel(np.exp(-2*-2.9756)),
        util.noise_kernel(np.exp(2*-0.2781))
)

y_kernel = util.summed_kernel(
        util.matern_kernel(np.exp(2.4624), np.exp(2*3.1776)),
        util.linear_kernel(np.exp(-2*-3.4571)),
        util.noise_kernel(np.exp(2*-0.3478))
)

def test1(debug_mode = []):
    # test of all functions in simulation with one or zero peds
    print "#################TEST4#################"
    print "# SIMULATION SIMPLE TEST"
    print "# ONE OR NONE PEDESTRIAN"
    print "#######################################"

    screen_size = (800,600)
    screen_name = "IGP unit test 1"
    frame_per_second = 2.5

    sim_env = simulation.Simulation_Environment(screen_size, screen_name, frame_per_second)
    sim_env.set_debug_mode(debug_mode)
    print ""
    print "--------------------------------------"
    print "test case for no pedestrian: "
    print "number of pedestrians: ", len(sim_env.pedestrian_list)
    print "current time : ", sim_env.current_time
    sim_env.run()


    parameters = {
            'ID': 3,
            'initial_position': np.array([100.,100.]),
            'goal_position': np.array([300.,200.]),
            'expected_speed': 60.,
            'sample_count': 100,
            'x_kernel': x_kernel,
            'y_kernel': y_kernel,
            'h': 20.,
            'alpha': 1.0,
            }
    pedestrian = simulation.Pedestrian(parameters, sim_env)
    sim_env.add_pedestrian(pedestrian)
    print "---------------------------------------"
    print "add first pedestrian"
    print "start time: ", sim_env.current_time
    print "initial position: ", np.array([100.,100.])
    sim_env.run()

    print ""
    print "---------------------------------------"
    print "remove first pedestrian"
    print "current time: ", sim_env.current_time
    sim_env.remove_pedestrian([3])
    sim_env.run()
    
    parameters = {
            'ID': 3,
            'initial_position': np.array([100.,100.]),
            'goal_position': np.array([102.,102.]),
            'expected_speed': 60.,
            'sample_count': 100,
            'x_kernel': x_kernel,
            'y_kernel': y_kernel,
            'h': 20.,
            'alpha': 1.0,
            }
    pedestrian = simulation.Pedestrian(parameters, sim_env)
    sim_env.add_pedestrian(pedestrian)
    print ""
    print "---------------------------------------"
    print "one step end test"
    print "add first pedestrian"
    print "start time: ", sim_env.current_time
    print "initial position: ", np.array([100.,100.])
    sim_env.run()


def test2(debug_mode = []):
    # test of all functions in simulation with one or zero peds
    print "#################TEST4#################"
    print "# SIMULATION SIMPLE TEST"
    print "# MORE PEDESTRIANS"
    print "#######################################"

    screen_size = (800,600)
    screen_name = "IGP unit test2"
    frame_per_second = 2.5

    sim_env = simulation.Simulation_Environment(screen_size, screen_name, frame_per_second)
    sim_env.set_debug_mode(debug_mode)
    print ""
    print "--------------------------------------"
    print "test case for two pedestrian: "

    parameters = {
            'ID': 3,
            'initial_position': np.array([100.,100.]),
            'goal_position': np.array([300.,200.]),
            'expected_speed': 60.,
            'sample_count': 100,
            'x_kernel': x_kernel,
            'y_kernel': y_kernel,
            'h': 20.,
            'alpha': 0.999,
            }
    pedestrian = simulation.Pedestrian(parameters, sim_env)
    sim_env.add_pedestrian(pedestrian)

    parameters = {
            'ID': 4,
            'initial_position': np.array([105.,105.]),
            'goal_position': np.array([305.,205.]),
            'expected_speed': 60.,
            'sample_count': 100,
            'x_kernel': x_kernel,
            'y_kernel': y_kernel,
            'h': 20.,
            'alpha': 0.999,
            }
    pedestrian = simulation.Pedestrian(parameters, sim_env)
    sim_env.add_pedestrian(pedestrian)
    print "---------------------------------------"
    print "add two pedestrians"
    print "start time: ", sim_env.current_time
    print "initial position for ID3: ", np.array([100.,100.])
    print "initial position for ID4: ", np.array([105.,105.])
    sim_env.run()

    print "---------------------------------------"
    print "remove pedestrian ID3"
    sim_env.remove_pedestrian([3])
    sim_env.run()
    # zero ped, one ped, two peds update and find next position funcs
    # also test edge case of pedestrian walking (only two ped)
    # ex: one ped start walking at its goal, a ped start walking when one ended



#test1([1,2,3,4])
test2([1,2,3,4])


