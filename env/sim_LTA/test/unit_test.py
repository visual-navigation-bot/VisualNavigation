import simulation
import numpy as np
import time


def test1(debug_mode = []): 
    print "#################TEST1################"
    print "# Add remove and calculate cross pedestrian value function test"
    print "######################################"
    sim = simulation.LTA((800, 600), 50)
    # set it to debug mode
    sim.set_debug_mode(debug_mode)

    parameters = {
            'ID' : 7,
            'lambda1' : 2.33,
            'lambda2' : 2.073,
            'sigma_d' : 0.361,
            'sigma_w' : 2.088,
            'beta' : 1.462,
            'alpha' : 0.730,
            'pixel2meters' : 0.2,
            'expected_speed': 6.5,
            'goal_position': np.array([500., 500.]),
            'initial_velocity': np.array([5., 0.]),
            'initial_position': np.array([100., 100.])
                }
    sim.add_ped(parameters)

    print "#########################"
    print "test1-1: one object, test calculate cross pedestrian value function"
    print ""
    print "should print:"
    print "pedestrian count : 1"
    print "pedestrian ID to index: 7 --> 0"
    print "pedestrian relative position: [[[0.  0.]]"
    print "all pedestrian velocity: [[5.  0.]]"
    print ""
    sim._calculate_cross_pedestrian_value()

    parameters = {
            'ID' : 4,
            'lambda1' : 2.33,
            'lambda2' : 2.073,
            'sigma_d' : 0.361,
            'sigma_w' : 2.088,
            'beta' : 1.462,
            'alpha' : 0.730,
            'pixel2meters' : 0.2,
            'expected_speed': 6.5,
            'goal_position': np.array([400., 500.]),
            'initial_velocity': np.array([3.4, 0.]),
            'initial_position': np.array([80., 140.])
                }
    sim.add_ped(parameters)

    print "#########################"
    print "test1-2: two object, test calculate cross pedestrian value function"
    print ""
    print "should print:"
    print "pedestrian count : 2"
    print "pedestrian ID to index: 7 --> 0, 4 --> 1"
    print "pedestrian relative position: "
    print "    ", np.array([[[0., 0.],[20., -40.]],[[-20., 40.],[0., 0.]]])
    print "all pedestrian velocity: [[5.  0.], [3.4  0.]]"
    print ""
    sim._calculate_cross_pedestrian_value()
    

    parameters = {
            'ID' : 3,
            'lambda1' : 2.33,
            'lambda2' : 2.073,
            'sigma_d' : 0.361,
            'sigma_w' : 2.088,
            'beta' : 1.462,
            'alpha' : 0.730,
            'pixel2meters' : 0.2,
            'expected_speed': 6.5,
            'goal_position': np.array([400., 500.]),
            'initial_velocity': np.array([3.4, 1.]),
            'initial_position': np.array([90., 190.])
                }
    sim.add_ped(parameters)

    print "#########################"
    print "test1-3: three object, test calculate cross pedestrian value function"
    print ""
    print "should print:"
    print "pedestrian count : 3"
    print "pedestrian ID to index: 7 --> 0, 4 --> 1, 3 --> 2"
    print "pedestrian relative position: "
    print "    ", np.array([[[0., 0.],[20., -40.],[10., -90.]],[[-20., 40.],[0., 0.],[-10., -50.]],[[-10.,90.],[10.,50.],[0.,0.]]])
    print "all pedestrian velocity: [[5.  0.], [3.4  0.], [3.4  1.]]"
    print ""
    sim._calculate_cross_pedestrian_value()


    sim._remove_pedestrian(4)
    print "#########################"
    print "test1-4: two object, test remove objects"
    print ""
    print "should print:"
    print "pedestrian count : 2"
    print "pedestrian ID to index: 7 --> 0, 3 --> 1"
    print "pedestrian relative position: "
    print "    ", np.array([[[0., 0.],[10., -90.]],[[-10.,90.],[0.,0.]]])
    print "all pedestrian velocity: [[5.  0.], [3.4  1.]]"
    print ""
    sim._calculate_cross_pedestrian_value()

    
    sim._remove_pedestrian(7)
    print "#########################"
    print "test1-5: one object, test remove objects" 
    print ""
    print "should print:"
    print "pedestrian count : 1"
    print "pedestrian ID to index: 3 --> 0"
    print "pedestrian relative position: [[[0.  0.]]"
    print "all pedestrian velocity: [[3.4  1.]]"
    print ""
    sim._calculate_cross_pedestrian_value()

    parameters = {
            'ID' : 5,
            'lambda1' : 2.33,
            'lambda2' : 2.073,
            'sigma_d' : 0.361,
            'sigma_w' : 2.088,
            'beta' : 1.462,
            'alpha' : 0.730,
            'pixel2meters' : 0.2,
            'expected_speed': 6.5,
            'goal_position': np.array([400., 500.]),
            'initial_velocity': np.array([3., 0.]),
            'initial_position': np.array([10., 10.])
                }
    sim.add_ped(parameters)


    sim._remove_pedestrian(5)
    sim._remove_pedestrian(3)
    print "#########################"
    print "test1-6: no object, test remove multiple objects"
    print ""
    print "should print:"
    print "pedestrian count : 0"
    print "pedestrian ID to index: "
    print "pedestrian relative position: [[]]"
    print "all pedestrian velocity: [[]]"
    print ""
    sim._calculate_cross_pedestrian_value()

# test RMSprop by using few simple functions
# test minimize_energy_velocity function
# test 3 imple test for run, move and display function

def brute_force_calculate_energy(ped, sim, velocity, debug_mode):
    # brute force calculate energy by for loops
    # return I, S, D separately
    l1 = ped.lambda1
    l2 = ped.lambda2
    alpha = ped.alpha
    beta = ped.beta
    sd = ped.sigma_d
    sw = ped.sigma_w
    p2m = ped.pixel2meters
    ID = ped.ID
    index = np.where(sim.ped_ID == ID)[0][0]

    vt = ped.velocity * p2m
    p = ped.position * p2m
    z = ped.goal_position * p2m
    u = ped.expected_speed * p2m
    v = velocity *p2m

    S = l1 * (u - np.linalg.norm(v)) ** 2
    D = - l2 * np.dot((z-p), v) / (np.linalg.norm(v) * np.linalg.norm(z-p))
    ped_count = len(sim.ped_list)
    E_sum = 0.

    if 2 in debug_mode:
        print "#########current ped index: ", index
    for i in range(ped_count):
        if i != index:
            ped2 = sim.ped_list[i]
            p2 = ped2.position * p2m
            v2 = ped2.velocity * p2m
            k = p - p2
            q = v - v2
            t = - np.dot(k, q) / np.linalg.norm(q) ** 2
            d = k + q * max(t, 0)        
            dsquare = np.linalg.norm(d)**2
            E = np.exp( -dsquare / (2 * sd**2))
            wd = np.exp(-np.linalg.norm(k)**2 / (2 * sw**2))
            cos = - np.dot(k, vt) / (np.linalg.norm(k) * np.linalg.norm(vt))
            wphi = ((1 + cos) / 2)**beta
            E_sum += E * wd * wphi

            if 2 in debug_mode:
                print "-----------------------"
                print "social energy to pedestrian index ", i
                print "wd: ", wd
                print "wphi: ", wphi
                print "k: ", k
                print "q: ", q
                print "d: ", d
                print "E: ", E
    I = E_sum
    if debug_mode:
        print "speed energy S: ", S
        print "direction energy D: ", D
        print "social energy I: ", I
        print "total energy E: ", I + S + D
    return I, S, D

def brute_force_energy_gradient(ped, sim, velocity, debug_mode):
    # brute force calculate energy gradient
    # return gradient I, S and D
    dv = 0.00001
    vchangex = np.array([velocity[0] + dv, velocity[1]])
    vchangey = np.array([velocity[0], velocity[1] + dv])

    E = brute_force_calculate_energy(ped, sim, velocity, debug_mode)
    Echangex = brute_force_calculate_energy(ped, sim, vchangex, debug_mode)
    Echangey = brute_force_calculate_energy(ped, sim, vchangey, debug_mode)
    gIx = (Echangex[0] - E[0]) / dv
    gIy = (Echangey[0] - E[0]) / dv

    gSx = (Echangex[1] - E[1]) / dv
    gSy = (Echangey[1] - E[1]) / dv

    gDx = (Echangex[2] - E[2]) / dv
    gDy = (Echangey[2] - E[2]) / dv

    gI = np.array([gIx, gIy])
    gS = np.array([gSx, gSy])
    gD = np.array([gDx, gDy])
    
    if 2 in debug_mode:
        print "gI: ", gI
        print "gS: ", gS
        print "gD: ", gD

    return gI, gS, gD



def test2(debug_mode = []):
    print "#################TEST2################"
    print "# ENERGY FUNCTION TEST: "
    print "# BRUTE FORCE SHOULD EQUAL TO VECTORIZED RESULT"
    print "######################################"

    sim = simulation.LTA((800, 600), 50)
    # set it to debug mode
    sim.set_debug_mode(debug_mode)

    parameters = {
            'ID' : 0,
            'lambda1' : 2.33,
            'lambda2' : 2.073,
            'sigma_d' : 0.361,
            'sigma_w' : 2.088,
            'beta' : 1.462,
            'alpha' : 0.730,
            'pixel2meters' : 0.2,
            'expected_speed': 6.5,
            'goal_position': np.array([505., 501.]),
            'initial_velocity': np.array([5.1, 0.]),
            'initial_position': np.array([103., 99.])
                }
    sim.add_ped(parameters)

    parameters = {
            'ID' : 1,
            'lambda1' : 2.33,
            'lambda2' : 2.073,
            'sigma_d' : 0.361,
            'sigma_w' : 2.088,
            'beta' : 1.462,
            'alpha' : 0.730,
            'pixel2meters' : 0.2,
            'expected_speed': 6.0,
            'goal_position': np.array([51., 106.]),
            'initial_velocity': np.array([4.5, 0.1]),
            'initial_position': np.array([124., 102.])
                }
    sim.add_ped(parameters)

    parameters = {
            'ID' : 2,
            'lambda1' : 2.33,
            'lambda2' : 2.073,
            'sigma_d' : 0.361,
            'sigma_w' : 2.088,
            'beta' : 1.462,
            'alpha' : 0.730,
            'pixel2meters' : 0.2,
            'expected_speed': 7.0,
            'goal_position': np.array([150., 151.]),
            'initial_velocity': np.array([3.0, 2.9]),
            'initial_position': np.array([101., 110.])
                }
    sim.add_ped(parameters)

    parameters = {
            'ID' : 3,
            'lambda1' : 2.33,
            'lambda2' : 2.073,
            'sigma_d' : 0.361,
            'sigma_w' : 2.088,
            'beta' : 1.462,
            'alpha' : 0.730,
            'pixel2meters' : 0.2,
            'expected_speed': 6.4,
            'goal_position': np.array([353., 252.]),
            'initial_velocity': np.array([2.0, 4.1]),
            'initial_position': np.array([96., 94.])
                }
    sim.add_ped(parameters)

    sim._calculate_cross_pedestrian_value()

    print "##############################"
    print "# FOUR PEDESTRIANS "
    print "##############################"

    for ped in sim.ped_list:
        velocity = np.array([5.5, 0.])
        print "Energy Test for Pedestrian ", ped.ID
        print "\nBRUTE FORCE CALCULATE ENERGY"
        I, S, D = brute_force_calculate_energy(ped, sim, velocity, debug_mode)
        print "E: ", I+S+D
        print "\nVECTORIZE CALCULATE ENERGY"
        E, gE = ped._energy_with_gradient(velocity)
        print "E: ", E
        print "========================="
        print ""

    sim._remove_pedestrian(1)
    sim._calculate_cross_pedestrian_value()

    print "##############################"
    print "# THREE PEDESTRIANS "
    print "##############################"
    for ped in sim.ped_list:
        velocity = np.array([5.5, 0.])
        print "Energy Test for Pedestrian ", ped.ID
        print "\nBRUTE FORCE CALCULATE ENERGY"
        I, S, D = brute_force_calculate_energy(ped, sim, velocity, debug_mode)
        print "E: ", I+S+D
        print "\nVECTORIZE CALCULATE ENERGY"
        E, gE = ped._energy_with_gradient(velocity)
        print "E: ", E
        print "========================="
        print ""

    sim._remove_pedestrian(2)
    sim._remove_pedestrian(0)
    sim._calculate_cross_pedestrian_value()

    print "##############################"
    print "# ONE PEDESTRIAN "
    print "##############################"

    for ped in sim.ped_list:
        velocity = np.array([5.5, 0.])
        print "Energy Test for Pedestrian ", ped.ID
        print "\nBRUTE FORCE CALCULATE ENERGY"
        I, S, D = brute_force_calculate_energy(ped, sim, velocity, debug_mode)
        print "E: ", I+S+D
        print "\nVECTORIZE CALCULATE ENERGY"
        E, gE = ped._energy_with_gradient(velocity)
        print "E: ", E
        print "========================="
        print ""


def test3(debug_mode = []):
    print "#################TEST3################"
    print "# GRADIENT ENERGY FUNCTION TEST: "
    print "# BRUTE FORCE SHOULD EQUAL TO VECTORIZED RESULT"
    print "######################################"

    sim = simulation.LTA((800, 600), 50)
    # set it to debug mode
    sim.set_debug_mode(debug_mode)

    parameters = {
            'ID' : 0,
            'lambda1' : 2.33,
            'lambda2' : 2.073,
            'sigma_d' : 0.361,
            'sigma_w' : 2.088,
            'beta' : 1.462,
            'alpha' : 0.730,
            'pixel2meters' : 0.2,
            'expected_speed': 6.5,
            'goal_position': np.array([505., 501.]),
            'initial_velocity': np.array([5.1, 0.]),
            'initial_position': np.array([103., 99.])
                }
    sim.add_ped(parameters)

    parameters = {
            'ID' : 1,
            'lambda1' : 2.33,
            'lambda2' : 2.073,
            'sigma_d' : 0.361,
            'sigma_w' : 2.088,
            'beta' : 1.462,
            'alpha' : 0.730,
            'pixel2meters' : 0.2,
            'expected_speed': 6.0,
            'goal_position': np.array([51., 106.]),
            'initial_velocity': np.array([4.5, 0.1]),
            'initial_position': np.array([124., 102.])
                }
    sim.add_ped(parameters)

    parameters = {
            'ID' : 2,
            'lambda1' : 2.33,
            'lambda2' : 2.073,
            'sigma_d' : 0.361,
            'sigma_w' : 2.088,
            'beta' : 1.462,
            'alpha' : 0.730,
            'pixel2meters' : 0.2,
            'expected_speed': 7.0,
            'goal_position': np.array([150., 151.]),
            'initial_velocity': np.array([3.0, 2.9]),
            'initial_position': np.array([101., 110.])
                }
    sim.add_ped(parameters)

    parameters = {
            'ID' : 3,
            'lambda1' : 2.33,
            'lambda2' : 2.073,
            'sigma_d' : 0.361,
            'sigma_w' : 2.088,
            'beta' : 1.462,
            'alpha' : 0.730,
            'pixel2meters' : 0.2,
            'expected_speed': 6.4,
            'goal_position': np.array([353., 252.]),
            'initial_velocity': np.array([2.0, 4.1]),
            'initial_position': np.array([96., 94.])
                }
    sim.add_ped(parameters)
    sim._calculate_cross_pedestrian_value()

    print "##############################"
    print "# FOUR PEDESTRIANS "
    print "##############################"
    for ped in sim.ped_list:
        velocity = np.array([5.5, 0.])
        print "Energy Test for Pedestrian ", ped.ID
        print "\nBRUTE FORCE CALCULATE ENERGY GRADIENT"
        gI, gS, gD = brute_force_energy_gradient(ped, sim, velocity, debug_mode)
        print "gE: ", gI + gS + gD
        print "\nVECTORIZE CALCULATE ENERGY GRADIENT"
        E, gE = ped._energy_with_gradient(velocity)
        print "gE: ", gE
        print "========================="
        print ""

    sim._remove_pedestrian(2)
    sim._calculate_cross_pedestrian_value()

    print "##############################"
    print "# THREE PEDESTRIANS "
    print "##############################"
    for ped in sim.ped_list:
        velocity = np.array([5.5, 0.])
        print "Energy Test for Pedestrian ", ped.ID
        print "\nBRUTE FORCE CALCULATE ENERGY GRADIENT"
        gI, gS, gD = brute_force_energy_gradient(ped, sim, velocity, debug_mode)
        print "gE: ", gI + gS + gD
        print "\nVECTORIZE CALCULATE ENERGY GRADIENT"
        E, gE = ped._energy_with_gradient(velocity)
        print "gE: ", gE
        print "========================="
        print ""

    sim._remove_pedestrian(0)
    sim._remove_pedestrian(1)
    sim._calculate_cross_pedestrian_value()

    print "##############################"
    print "# ONE PEDESTRIAN "
    print "##############################"
    for ped in sim.ped_list:
        velocity = np.array([5.5, 0.])
        print "Energy Test for Pedestrian ", ped.ID
        print "\nBRUTE FORCE CALCULATE ENERGY GRADIENT"
        gI, gS, gD = brute_force_energy_gradient(ped, sim, velocity, debug_mode)
        print "gE: ", gI + gS + gD
        print "\nVECTORIZE CALCULATE ENERGY GRADIENT"
        E, gE = ped._energy_with_gradient(velocity)
        print "gE: ", gE
        print "========================="
        print ""


def test4():
    print "#################TEST4################"
    print "# RMSprop SIMPLE TEST: "
    print "######################################"

    print "############################"
    print "# PART 1: |v|**2"
    print "############################"

    def energy_with_gradient1(x):
        return x[0]**2 + x[1]**2 + 10, np.array([2*x[0], 2*x[1]])

    parameters = {}
    parameters['gamma'] = 0.999
    parameters['alpha'] = 0.1 #0.001
    parameters['epsilon'] = 10**(-8)

    initial_velocity = np.array([10., 10.])
    energy_list, minimize_energy_velocity = simulation.RMSprop(
            initial_velocity, energy_with_gradient1, parameters)

    print "energy decay process: "
    for energy in energy_list:
        print energy
    print "velocity that minimize energy: ", minimize_energy_velocity
    print "steps:", len(energy_list)


def test5(debug_mode = []):
    print "#################TEST5################"
    print "# RMSpop MINIMIZE ENERGY TEST"
    print "######################################"

    sim = simulation.LTA((800, 600), 2.5)
    sim.set_debug_mode(debug_mode)
    parameters = {
            'ID' : 0,
            'lambda1' : 2.33,
            'lambda2' : 2.073,
            'sigma_d' : 0.361,
            'sigma_w' : 2.088,
            'beta' : 1.462,
            'alpha' : 0.730,
            'pixel2meters' : 0.2,
            'expected_speed': 6.5,
            'goal_position': np.array([505., 101.]),
            'initial_velocity': np.array([5.1, 0.]),
            'initial_position': np.array([103., 99.])
                }
    sim.add_ped(parameters)

    parameters = {
            'ID' : 1,
            'lambda1' : 2.33,
            'lambda2' : 2.073,
            'sigma_d' : 0.361,
            'sigma_w' : 2.088,
            'beta' : 1.462,
            'alpha' : 0.730,
            'pixel2meters' : 0.2,
            'expected_speed': 6.0,
            'goal_position': np.array([51., 106.]),
            'initial_velocity': np.array([- 4.5, 0.1]),
            'initial_position': np.array([109., 98.])
                }
    sim.add_ped(parameters)

    parameters = {
            'ID' : 2,
            'lambda1' : 2.33,
            'lambda2' : 2.073,
            'sigma_d' : 0.361,
            'sigma_w' : 2.088,
            'beta' : 1.462,
            'alpha' : 0.730,
            'pixel2meters' : 0.2,
            'expected_speed': 7.0,
            'goal_position': np.array([150., 151.]),
            'initial_velocity': np.array([3.0, 2.9]),
            'initial_position': np.array([101., 110.])
                }
    sim.add_ped(parameters)

    parameters = {
            'ID' : 3,
            'lambda1' : 2.33,
            'lambda2' : 2.073,
            'sigma_d' : 0.361,
            'sigma_w' : 2.088,
            'beta' : 1.462,
            'alpha' : 0.730,
            'pixel2meters' : 0.2,
            'expected_speed': 6.4,
            'goal_position': np.array([353., 252.]),
            'initial_velocity': np.array([2.0, 4.1]),
            'initial_position': np.array([96., 94.])
                }
    sim.add_ped(parameters)
    sim._calculate_cross_pedestrian_value()
    
    print "##############################"
    print "# FOUR PEDESTRIANS "
    print "##############################"
    for ped in sim.ped_list:
        print "optimal velocity for Pedestrian ", ped.ID
        optimal_velocity = ped._minimize_energy_velocity()
        print optimal_velocity
        print "========================"
        print ""

    sim._remove_pedestrian(1)
    sim._calculate_cross_pedestrian_value()
    
    print "##############################"
    print "# THREE PEDESTRIANS "
    print "##############################"
    for ped in sim.ped_list:
        print "optimal velocity for Pedestrian ", ped.ID
        optimal_velocity = ped._minimize_energy_velocity()
        print optimal_velocity
        print "========================"
        print ""

    sim._remove_pedestrian(0)
    sim._remove_pedestrian(3)
    sim._calculate_cross_pedestrian_value()
    
    print "##############################"
    print "# ONE PEDESTRIAN "
    print "##############################"
    for ped in sim.ped_list:
        print "optimal velocity for Pedestrian ", ped.ID
        optimal_velocity = ped._minimize_energy_velocity()
        print "optimal velocity: ", optimal_velocity
        print "========================"
        print ""

# debug mode: 
# 1 --> display cross pedestrian value
# 2 --> display energy calculation detail
# 3 --> display minimize energy process

