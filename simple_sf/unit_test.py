import simulation as sim
import pygame
import numpy as np
import random

# this is the unit test file of all functions in simulation.py
def test1():
    """
    test distance matrix calculation, party formulation variable,
    id record method, add_object method and remove_object method
    """
    print "################TEST1##################"


    sim_env = sim.Sim('Pedestrian Simulation', (800,600))
    sim_env._get_dist_matrix()
    print "test1-1: no object dist_matrix calculation"
    print "should print out []:"
    print sim_env.PARTY_DIS
    print ""

    print "test1-2: no object party v calculation"
    print "should print out []"
    print sim_env.PARTY_V
    print ""

    print "test1-3: no object party r calculation"
    print "should print out []"
    print sim_env.PARTY_R
    print ""

    print "test1-4: no object friend id"
    print "should print out []"
    print sim_env.FRIEND
    print ""

    print "test1-5: pedestian id dictionary confirmation"
    print "should print out {}"
    print sim_env.PID
    print ""

    def dist_obstacle(s):
        """
        calculate closest vector distant from the ped to obstacle
        Input:
            s: np.1darray; the location of the pedestrian
        """
        if s[1] - 100 < 500 - s[1]:
            return np.array([0., s[1] - 100.])
        if s[1] - 100 > 500 - s[1]:
            return np.array([0., s[1] - 500.])
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
             'dist_obs_func': dist_obstacle,
             'pid': 5
             }
    sim_env.add_object('ped', param)
    sim_env._get_dist_matrix()
    print "test1-6: single object dist_matrix calculation"
    print "should print out [[[0., 0.]]]:"
    print sim_env.PARTY_DIS
    print ""

    print "test1-7: no object party v calculation"
    print "should print out [[0.]]"
    print sim_env.PARTY_V
    print ""

    print "test1-8: no object party r calculation"
    print "should print out [[0.]]"
    print sim_env.PARTY_R
    print ""

    print "test1-9: no object friend id"
    print "should print out [[True]]"
    print sim_env.FRIEND
    print ""

    print "test1-10: pedestian id dictionary confirmation"
    print "should print out {5:0}"
    print sim_env.PID
    print ""

    param = {
             'init_s': np.array([130., 200.]),
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
             'dist_obs_func': dist_obstacle,
             'pid': 3
             }
    sim_env.add_object('ped', param)
    sim_env._get_dist_matrix()
    print "test1-11: double object dist_matrix calculation"
    print 'should print out [[[0., 0.], [30., 50.]], \n \
            [[-30., -50.], [0., 0.]]]:'
    print sim_env.PARTY_DIS
    print ""

    print "test1-12: no object party v calculation"
    print "should print out [[0., 100.], [100., 0.]]"
    print sim_env.PARTY_V
    print ""

    print "test1-13: no object party r calculation"
    print "should print out [[0., 50.], [50., 0.]]"
    print sim_env.PARTY_R
    print ""

    print "test1-14: no object friend id"
    print "should print out [[True, True], [True, True]]"
    print sim_env.FRIEND
    print ""

    print "test1-15: pedestian id dictionary confirmation"
    print "should print out {5:0, 3:1}"
    print sim_env.PID
    print ""

    param = {
             'init_s': np.array([200., 300.]),
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
             'V_others': np.array([200., 300.]),
             'R_others': np.array([60., 70.]),
             'dist_obs_func': dist_obstacle,
             'pid': 1
             }
    sim_env.add_object('ped', param)
    sim_env._get_dist_matrix()
    print "test1-16: triple object dist_matrix calculation"
    print 'should print out [[0.,0.],[30.,50.],[100., 150.]]\n \
            [[-30.,-50.],[0.,0.],[70., 100.]]\n\
            [[-100.,-150.],[-70.,-100.],[0.,0.]]'
    print sim_env.PARTY_DIS
    print ""

    print "test1-17: no object party v calculation"
    print "should print out [[0., 100., 200.], \n\
            [100., 0., 300.],\n[200.,300.,0.]]"
    print sim_env.PARTY_V
    print ""

    print "test1-18: no object party r calculation"
    print "should print out [[0., 50., 60.], \n\
            [50., 0., 70.], \n[60., 70., 0.]]"
    print sim_env.PARTY_R
    print ""

    print "test1-19: no object friend id"
    print "should print out [[True, True, False], \n\
            [True, True, True], \n[False, True, True]]"
    print sim_env.FRIEND
    print ""

    print "test1-20: pedestian id dictionary confirmation"
    print "should print out {5:0, 3:1, 1:2}"
    print sim_env.PID
    print ""

    #test of remove function
    sim_env.remove_pedestrians([3])
    sim_env._get_dist_matrix()
    print "test1-21: triple object dist_matrix calculation"
    print 'should print out [[0.,0.],[100., 150.]], [[-100.,-150.],[0.,0.]]'
    print sim_env.PARTY_DIS
    print ""

    print "test1-22: remove pedestrian function on party r result"
    print "should print out [[0., 60.],[60., 0.]]"
    print sim_env.PARTY_R
    print ""

    print "test1-23: remove pedestrian function on pedestian id"
    print "should print out {5:0, 1:1}"
    print sim_env.PID
    print ""

    print "test1-24: remove pedestrian function on pedestrian list"
    print "should print out 5 , 1"
    for ped in sim_env.pedestrians:
        print ped.pid

    sim_env.remove_pedestrians([1])
    sim_env._get_dist_matrix()
    print "test1-25: triple object dist_matrix calculation"
    print 'should print out [[0.,0.]]'
    print sim_env.PARTY_DIS
    print ""

    print "test1-26: remove pedestrian function on party r result"
    print "should print out [[0.]]"
    print sim_env.PARTY_R
    print ""

    print "test1-27: remove pedestrian function on pedestian id"
    print "should print out {5:0}"
    print sim_env.PID
    print ""

    print "test1-28: remove pedestrian function on pedestrian list"
    print "should print out 5"
    for ped in sim_env.pedestrians:
        print ped.pid

def obstacle_force_check(sim_env):
    """
    given a simulation environment
    check the correctness of obstacle force
    """
    def obstacle_force(ped):
        # using the pedestrian variables to calculate energy in s
        dx = np.array([0.001, 0])
        dy = np.array([0, 0.001])

        r2x = ped.dist_obs_func(ped.s + dx)
        r1x = ped.dist_obs_func(ped.s)
        U2x = ped.V_obs * np.exp(-np.linalg.norm(r2x) / ped.R_obs)
        U1x = ped.V_obs * np.exp(-np.linalg.norm(r1x) / ped.R_obs)

        Fx = - (U2x - U1x) / 0.001

        r2y = ped.dist_obs_func(ped.s + dy)
        r1y = ped.dist_obs_func(ped.s)
        U2y = ped.V_obs * np.exp(-np.linalg.norm(r2y) / ped.R_obs)
        U1y = ped.V_obs * np.exp(-np.linalg.norm(r1y) / ped.R_obs)

        Fy = - (U2y - U1y) / 0.001
        return np.array([Fx, Fy])

    num = 0
    for ped in sim_env.pedestrians:
        print "(1) pedestrian" +str(num) +" obstacle force check"
        ped = sim_env.pedestrians[0]
        print "numerical value: ", obstacle_force(ped)
        print "function value: ", ped.obstacle_force()
        print ""
        num += 1

def repulsive_force_check(sim_env):
    """
    given a simulation environment
    check the correctness of repulsive force and sight angle
    """
    def repulsive_force(ped, sim):
        # using the pedestrian and simulation environment
        # to calculate energy in s
        ped_idx = sim.PID[ped.pid]
        f_sum = np.array([0., 0.])
        for idx in range(len(sim.pedestrians)):
            if idx != ped_idx:
                V = sim.PARTY_V[idx][ped_idx]
                R = sim.PARTY_R[idx][ped_idx]
                s_this = ped.s
                s_that = sim.pedestrians[idx].s

                dx = np.array([0.001, 0.])
                dy = np.array([0., 0.001])

                b2x = np.linalg.norm(s_this - s_that + dx)
                b1x = np.linalg.norm(s_this - s_that)
                V2x = V * np.exp(-b2x / R)
                V1x = V * np.exp(-b1x / R)
                Fx =  - (V2x - V1x) / 0.001

                b2y = np.linalg.norm(s_this - s_that + dy)
                b1y = np.linalg.norm(s_this - s_that)
                V2y = V * np.exp(-b2y / R)
                V1y = V * np.exp(-b1y / R)
                Fy = - (V2y - V1y) / 0.001

                f= np.array([Fx, Fy])

                norm_ds = np.linalg.norm(s_that - s_this)
                norm_v = np.linalg.norm(ped.v)
                base_line = norm_ds*norm_v*np.cos(ped.sight_angle)
                if np.dot((s_that - s_this), ped.v) < base_line:
                    f_sum += f * ped.sight_const
                else:
                    f_sum += f
        return f_sum

    ped_num = len(sim_env.pedestrians)
    sim_env._get_dist_matrix()
    # update pedestrians and display them
    num = 0
    for idx in range(ped_num):
        # relative information to other pedestrians
        other_dist = np.delete(sim_env.PARTY_DIS[idx], idx, 0)
        other_V = np.delete(sim_env.PARTY_V[idx], idx, 0)
        other_R = np.delete(sim_env.PARTY_R[idx], idx, 0)
        
        print "(2) pedestrian" + str(num) + "  repulsive force and sight angle check"
        ped = sim_env.pedestrians[idx]
        print "numerical value: ", repulsive_force(ped, sim_env)
        print "function value: ", \
                ped.repulsive_force(other_dist, other_V, other_R) 
        print ""
        num += 1
    
def maximum_velocity_check(sim_env):
    """
    given a simulation environment
    check the correctness of maximum velocity limiation
    """
    ped_num = len(sim_env.pedestrians)
    for idx in range(ped_num):
        ped = sim_env.pedestrians[idx]
        ped.v = np.array([ped.vmax * 1.2, 0])

    sim_env._get_dist_matrix()
    # update pedestrians and display them
    num = 0
    for idx in range(ped_num):
        # relative information to other pedestrians
        other_dist = np.delete(sim_env.PARTY_DIS[idx], idx, 0)
        other_V = np.delete(sim_env.PARTY_V[idx], idx, 0)
        other_R = np.delete(sim_env.PARTY_R[idx], idx, 0)

        print "(3) pedestrian" + str(num) + " maximum velocity check"
        ped = sim_env.pedestrians[idx]
        ped.move(other_dist, other_V, other_R, 0.1)
        print "real v: ", ped.v
        print "max v: ",np.array([ped.vmax, 0.])
        print ""
        num += 1

def test2():
    """
    test target force, obstace force, repulsive force, sight angle
    max velocity correctness
    """
    print "################TEST2##################"

    sim_env = sim.Sim('Pedestrian Simulation', (800, 600))
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
            return np.array([0., s[1] - 100.])
        if s[1] - 100 > 500 - s[1]:
            return np.array([0., s[1] - 500.])
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
             'dist_obs_func': dist_obstacle,
             'pid': 0
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
             'dist_obs_func': dist_obstacle,
             'pid': 6
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
             'dist_obs_func': dist_obstacle,
             'pid': 4
             }
    sim_env.add_object('ped', param)
    print "test2-1: three pedestrians obstacle, repulsive force and maximum velocity check"
    obstacle_force_check(sim_env)
    repulsive_force_check(sim_env)
    maximum_velocity_check(sim_env)

    print "test2-2: two pedestrians obstacle, repulsive force and maximum velocity check"
    sim_env.remove_pedestrians([6])
    obstacle_force_check(sim_env)
    repulsive_force_check(sim_env)
    maximum_velocity_check(sim_env)

    print "test2-2: one pedestrians obstacle, repulsive force and maximum velocity check"
    sim_env.remove_pedestrians([0])
    obstacle_force_check(sim_env)
    maximum_velocity_check(sim_env)


def test3():
    """
    test simple run function 
    """
    print "################TEST3##################"

    sim_env = sim.Sim('Pedestrian Simulation', (800, 600))
    """
    adding obstacles
    """
    param = {
             'start': np.array([100,100]), 
             'end': np.array([700,100])
             }
    sim_env.add_object('obs', param)

    param = {
             'start': np.array([100,500]), 
             'end': np.array([700,500])
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
            return np.array([0., s[1] - 100.])
        if s[1] - 100 > 500 - s[1]:
            return np.array([0., s[1] - 500.])
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
             'dist_obs_func': dist_obstacle,
             'pid': 0
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
             'dist_obs_func': dist_obstacle,
             'pid': 6
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
             'dist_obs_func': dist_obstacle,
             'pid': 4
             }
    sim_env.add_object('ped', param)
    sim_env.run()


def test4():
    """
    test auto terminates the file
    """
    print "################TEST4##################"
    print "test4 test of custom auto terminate the file \n\
            test of variables testing\n\
            test of clock tick function and reset screen function"
    screen_size = (800, 600)
    sim_env = sim.Sim('Pedestrian Simulation', screen_size)
    """
    adding obstacles
    """
    param = {
             'start': np.array([100,100]), 
             'end': np.array([700,100])
             }
    sim_env.add_object('obs', param)

    param = {
             'start': np.array([100,500]), 
             'end': np.array([700,500])
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
            return np.array([0., s[1] - 100.])
        if s[1] - 100 > 500 - s[1]:
            return np.array([0., s[1] - 500.])
        return np.array([100000, 100000])

    param = {
             'init_s': np.array([100., 150.]),
             'exp_s': np.array([700., 150.]),
             'v0': 100,
             'vmax': 130,
             'init_v': np.array([100., 0.]),
             'tau': 0.5,
             'V_obs': 20000.,
             'R_obs': 500.,
             'sight_angle': np.pi / 2, 
             'sight_const': 0.5,
             'friend': np.array([]),
             'V_others': np.array([]),
             'R_others': np.array([]),
             'dist_obs_func': dist_obstacle,
             'pid': 0
             }
    sim_env.add_object('ped', param)

    param = {
             'init_s': np.array([100., 200.]),
             'exp_s': np.array([700., 200.]),
             'v0': 100,
             'vmax': 130,
             'init_v': np.array([100., 0.]),
             'tau': 0.5,
             'V_obs': 20000.,
             'R_obs': 500.,
             'sight_angle': np.pi / 2, 
             'sight_const': 0.5,
             'friend': np.array([True]),
             'V_others': np.array([10000.]),
             'R_others': np.array([30.]),
             'dist_obs_func': dist_obstacle,
             'pid': 6
             }
    sim_env.add_object('ped', param)


    def is_arrival(ped, sim_env):
        """
        check if the pedestrian has accomplish his/her tasks
        Input:
            ped: Ped; the pedestrian we want to check
            sim_env: Sim; the simulation environment
        Return:
            is_arrival: boolean; is the pedestrian arrives its destination
        """
        time_step = sim_env.TDIFF
        vmax = ped.vmax
        exp_s = ped.exp_s
        s = ped.s

        if np.linalg.norm(s - exp_s) < time_step * vmax * 5:
            return True
        else:
            return False


    running = True
    while running:
        sim_env.clock_tick()
        sim_env.reset_screen()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        for ped in sim_env.pedestrians:
            if is_arrival(ped, sim_env):
                pid = ped.pid
                sim_env.remove_pedestrians([pid])
        if len(sim_env.pedestrians) == 0:
            running = False

        sim_env.move()
        sim_env.display()
        pygame.display.flip()


#test1()
#test2()
test3()
#test4()

