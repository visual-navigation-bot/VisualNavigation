import simulation as sim
import pygame
import numpy as np
import random
import time

# This is tests fancy version tests
# case 1: one time multiple human different direction walking through a corridor
# case 2: continuous multiple human different direction walking through a corridor
# case 3: evacuation test case
def is_arrival(ped, sim_env):
    # check if the pedestrian has accomplish arrived its destination
    time_step = sim_env.TDIFF
    vmax = ped.vmax
    exp_s = ped.exp_s
    s = ped.s

    if np.linalg.norm(s - exp_s) < time_step * vmax * 5:
        return True
    else:
        return False

def obstacle_constraint(ped):
    # if the pedestrian is almost walking out of the obstacle
    # bring it back and constrains its location and speed
    margin = 5.
    if ped.s[1] < 100. + margin:
        ped.s[1] = 100. + margin
        norm_v = np.linalg.norm(ped.v)
        if ped.v[0] >= 0.:
            ped.v = np.array([norm_v, 0])
        else:
            ped.v = np.array([-norm_v, 0])
    if ped.s[1] > 500. - margin:
        ped.s[1] = 500. - margin
        norm_v = np.linalg.norm(ped.v)
        if ped.v[0] >= 0.:
            ped.v = np.array([norm_v, 0])
        else:
            ped.v = np.array([-norm_v, 0])
            



def test_corridor():
    """
    Environment: One direction corridor

    Pedestrians: walking two ways, but variables randomly set
                 the initial setted pedestrians will still be walking around
    """

    print "Simple Corridor Test Case"
    num_ped = random.randint(0, 10)
    print "number of pedestrians: ", num_ped

    screen_size = (800, 600)
    sim_env = sim.Sim('Pedestrian Simulation', screen_size)

    """
    adding obstacles
    """
    param = {
             'start': np.array([0,100]), 
             'end': np.array([800,100])
             }
    sim_env.add_object('obs', param)
    param = {
             'start': np.array([0,500]), 
             'end': np.array([800,500])
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
        return np.array([0., random.uniform(500 ,1000) * random.choice([1, -1])])

    for pid in range(num_ped):
        right_direction = random.randint(0, 1)

        v0 = random.uniform(60., 130.)
        vmax = 1.3 * v0
        theta_v = random.uniform(-np.pi / 6, np.pi / 6)
        tau = random.uniform(0.4, 0.6)
        V_obs = random.uniform(20000., 30000.)
        R_obs = random.uniform(100., 500.)
        sight_angle = random.uniform(np.pi / 4, np.pi / 2)
        sight_const = random.uniform(0.4, 1.0)
        friend = np.array([])
        V_others = np.array([])
        R_others = np.array([])
        for i in range(pid):
            friend = np.append(friend, random.choice([True, False]))
            V_others = np.append(V_others, random.uniform(5000., 100000.))
            R_others = np.append(R_others, random.uniform(20., 50.))

        if right_direction:
            init_s = np.array([random.uniform(-300., -100.), random.uniform(130., 470.)])
            exp_s = np.array([900., init_s[1]])
            init_v = np.array([v0 * np.cos(theta_v), v0 * np.sin(theta_v)])
        else:
            init_s = np.array([random.uniform(900., 1100.), random.uniform(130., 470.)])
            exp_s = np.array([-100., init_s[1]])
            init_v = np.array([- v0 * np.cos(theta_v), v0 * np.sin(theta_v)])

        param = {
                 'init_s': init_s,
                 'exp_s': exp_s,
                 'v0': v0,
                 'vmax': vmax,
                 'init_v': init_v,
                 'tau': tau,
                 'V_obs': V_obs,
                 'R_obs': R_obs,
                 'sight_angle': sight_angle, 
                 'sight_const': sight_const,
                 'friend': friend,
                 'V_others': V_others,
                 'R_others': R_others,
                 'dist_obs_func': dist_obstacle,
                 'pid': pid
                 }
        sim_env.add_object('ped', param)


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
        for ped in sim_env.pedestrians:
            obstacle_constraint(ped)
        sim_env.display()
        pygame.display.flip()

def test_continuous_corridor():
    """
    Environment: One direction corridor

    Pedestrians: walking two ways, but variables randomly set
                 there will continuous be pedestrians walking in and out
    """
    print "Continous Corridor Test Case"
    num_ped = random.randint(1, 7)
    maximum_num_ped = 20
    # there will be a pedestrian walking in every this amount of time
    time_walkin = [1., 2.5]

    print "Initial number of pedestrians: ", num_ped
    print "maximum capacity of pedestrians: ", maximum_num_ped

    screen_size = (800, 600)
    sim_env = sim.Sim('Pedestrian Simulation', screen_size)
    """
    adding obstacles
    """
    param = {
             'start': np.array([0,100]), 
             'end': np.array([800,100])
             }
    sim_env.add_object('obs', param)
    param = {
             'start': np.array([0,500]), 
             'end': np.array([800,500])
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
        return np.array([0., random.uniform(500 ,1000) * random.choice([1, -1])])

    for pid in range(num_ped):
        right_direction = random.randint(0, 1)

        v0 = random.uniform(60., 130.)
        vmax = 1.3 * v0
        theta_v = random.uniform(-np.pi / 6, np.pi / 6)
        tau = random.uniform(0.4, 0.6)
        V_obs = random.uniform(20000., 30000.)
        R_obs = random.uniform(100., 500.)
        sight_angle = random.uniform(np.pi / 4, np.pi / 2)
        sight_const = random.uniform(0.4, 1.0)
        friend = np.array([])
        V_others = np.array([])
        R_others = np.array([])
        for i in range(pid):
            friend = np.append(friend, random.choice([True, False]))
            V_others = np.append(V_others, random.uniform(5000., 100000.))
            R_others = np.append(R_others, random.uniform(20., 50.))

        if right_direction:
            init_s = np.array([random.uniform(-300., -100.), random.uniform(130., 470.)])
            exp_s = np.array([900., init_s[1]])
            init_v = np.array([v0 * np.cos(theta_v), v0 * np.sin(theta_v)])
        else:
            init_s = np.array([random.uniform(900., 1100.), random.uniform(130., 470.)])
            exp_s = np.array([-100., init_s[1]])
            init_v = np.array([- v0 * np.cos(theta_v), v0 * np.sin(theta_v)])

        param = {
                 'init_s': init_s,
                 'exp_s': exp_s,
                 'v0': v0,
                 'vmax': vmax,
                 'init_v': init_v,
                 'tau': tau,
                 'V_obs': V_obs,
                 'R_obs': R_obs,
                 'sight_angle': sight_angle, 
                 'sight_const': sight_const,
                 'friend': friend,
                 'V_others': V_others,
                 'R_others': R_others,
                 'dist_obs_func': dist_obstacle,
                 'pid': pid
                 }
        sim_env.add_object('ped', param)

    # the next pedestrian id
    next_pid = num_ped

    last_walkin_time = time.time()
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
        for ped in sim_env.pedestrians:
            obstacle_constraint(ped)


        # add a new pedestrian while some time passes    
        if time.time() - last_walkin_time > random.uniform(time_walkin[0], time_walkin[1]):
            num_ped = len(sim_env.pedestrians)
            if num_ped < maximum_num_ped:
                # add a new pedestrian
                right_direction = random.randint(0, 1)

                v0 = random.uniform(60., 130.)
                vmax = 1.3 * v0
                theta_v = random.uniform(-np.pi / 6, np.pi / 6)
                tau = random.uniform(0.4, 0.6)
                V_obs = random.uniform(20000., 30000.)
                R_obs = random.uniform(100., 500.)
                sight_angle = random.uniform(np.pi / 4, np.pi / 2)
                sight_const = random.uniform(0.4, 1.0)
                friend = np.array([])
                V_others = np.array([])
                R_others = np.array([])
                for i in range(num_ped):
                    friend = np.append(friend, random.choice([True, False]))
                    V_others = np.append(V_others, random.uniform(5000., 100000.))
                    R_others = np.append(R_others, random.uniform(20., 50.))

                if right_direction:
                    init_s = np.array([random.uniform(-300., -100.), random.uniform(130., 470.)])
                    exp_s = np.array([900., init_s[1]])
                    init_v = np.array([v0 * np.cos(theta_v), v0 * np.sin(theta_v)])
                else:
                    init_s = np.array([random.uniform(900., 1100.), random.uniform(130., 470.)])
                    exp_s = np.array([-100., init_s[1]])
                    init_v = np.array([- v0 * np.cos(theta_v), v0 * np.sin(theta_v)])

                param = {
                         'init_s': init_s,
                         'exp_s': exp_s,
                         'v0': v0,
                         'vmax': vmax,
                         'init_v': init_v,
                         'tau': tau,
                         'V_obs': V_obs,
                         'R_obs': R_obs,
                         'sight_angle': sight_angle, 
                         'sight_const': sight_const,
                         'friend': friend,
                         'V_others': V_others,
                         'R_others': R_others,
                         'dist_obs_func': dist_obstacle,
                         'pid': next_pid
                         }
                next_pid += 1
                last_walkin_time = time.time()
                sim_env.add_object('ped', param)
                print "new pedestrian join!"
                print "currently number of pedestrians: ", len(sim_env.pedestrians)

        sim_env.display()
        pygame.display.flip()

def test_evacuation():
    pass

#test_corridor()
test_continuous_corridor()
