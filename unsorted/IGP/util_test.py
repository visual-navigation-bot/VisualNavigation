import util
import matplotlib.pyplot as plt
import numpy as np
import pygame


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


def test1():
    # test Path_History initialize function, add_history function, sliced history function
    print "#################TEST1#################"
    print "# PATH HISTORY CLASS TEST"
    print "#######################################"

    init_time = 12
    init_position = np.array([20.,4.])
    velocity = np.array([1.,2.])
    ph = util.Path_History(init_time, init_position)

    position = init_position
    time = init_time
    time_count = 20

    for i in range(time_count):
        time += 1
        position = position + velocity
        ph.add_history(time, position)

    plt.figure(1)
    plt.clf()
    plt.title('path history test')
    ph.plot('o')

    whole_slice = ph.sliced_history(12)
    whole_slice.plot('x')
    part_slice = ph.sliced_history(20)
    part_slice.plot('^')

    past_time = np.arange(10, 30)
    init_position = np.array([10, 10])
    past_path = np.outer(past_time, velocity) + init_position
    ph = util.Path_History(past_time, past_path)

    plt.figure(2)
    plt.clf()
    plt.title('path history directly initiate test')
    ph.plot('o')

    whole_slice = ph.sliced_history(1)
    whole_slice.plot('x')
    part_slice = ph.sliced_history(29)
    part_slice.plot('^')

    plt.show()

def test2():
    # test the kernel functions
    print "#################TEST2#################"
    print "# KERNEL TEST"
    print "#######################################"
    sq = util.square_exponential_kernel(1., .5)
    v1 = np.array([[1.]])
    v2 = np.array([[1.]])
    print "test case 1: length 1 vector"
    print "should print: ",np.array([[0.5]])
    print sq(v1, v2, "train")
    print "----------------------------"

    v1 = np.array([[1.]])
    v2 = np.array([[1.],[2.]])
    print "test case 2: length 1 vector * length 2 vector"
    covariance = np.array([[.5, .5 * np.exp(-0.5)]])
    print "should print: ", covariance
    print sq(v1, v2, "train")
    print "----------------------------"

    v1 = np.array([[1.], [2.]])
    v2 = np.array([[1.], [2.]])
    print "test case 3: length 2 vector * length 2 vector"
    covariance = np.array([[.5, .5 * np.exp(-0.5)], [.5*np.exp(-0.5), .5]])
    print "should print: ", covariance
    print sq(v1, v2, "train")
    print "----------------------------"

    matern = util.matern_kernel(1., 1.)
    print "test case 4: length 2 vector * length 2 vector for matern kernel"
    print matern(v1, v2, "train")
    print "----------------------------"

    linear = util.linear_kernel(1.)
    print "test case 5: length 2 vector * length 2 vector for linear kernel"
    print linear(v1, v2, "train")
    print "----------------------------"

    noise = util.noise_kernel(.3)
    print "test case 6: length 2 vector * length 2 vector for noise kernel"
    print "training version: "
    print noise(v1, v2, "train")
    print "testing version: "
    print noise(v1, v2, "test")
    print "crossing version: "
    print noise(v1, v2, "cross")
    print "----------------------------"

    v1 = np.array([[1.]])
    print "test case 7: length 1 vector * length 2 vector for matern kernel"
    print matern(v1, v2, "train")
    print "----------------------------"

    print "test case 8: length 1 vector * length 2 vector for linear kernel"
    print linear(v1, v2, "train")
    print "----------------------------"

    summed_kernel = util.summed_kernel(
        util.matern_kernel(1., 1.),
        util.linear_kernel(1.),
        util.noise_kernel(.3)
        )
    print "test case 9: summed kernel"
    m = matern(v1, v2, "cross")
    l = linear(v1, v2, "cross")
    n = noise(v1, v2, "cross")
    print "should print out: ", m+l+n
    print summed_kernel(v1, v2, "cross")
    print "---------------------------"
    v1 = np.array([[1.],[2.]])
    m = matern(v1, v2, "train")
    l = linear(v1, v2, "train")
    n = noise(v1, v2, "train")
    print "should print out: ", m+l+n
    print summed_kernel(v1, v2, "train")
    print "---------------------------"

def test3(debug_mode = []):
    # test the gaussian process functions
    print "#################TEST3#################"
    print "# GAUSSIAN PROCESS TEST"
    print "#######################################"
    gp = util.Gaussian_Process(x_kernel)
    gp.set_debug_mode(debug_mode)

    gp.refresh()
    x_train = np.array([[0.]])
    y_train = np.array([[0.]])
    gp.add_train_set(x_train, y_train)

    x_test = np.arange(1,10).reshape(-1,1)
    gp.add_test_set(x_test)
    sample = gp.sample(10)
    plt.figure(1)
    plt.title('one train set, lots of test set')
    plt.plot(x_train, y_train, 'o')
    plt.plot(x_test, sample, 'o')

    ##########################################
    gp.refresh()
    x_train = np.arange(0, 10).reshape(-1,1)
    y_train = x_train * 2 + 2
    gp.add_train_set(x_train, y_train)

    x_test = np.arange(10,20).reshape(-1,1)
    gp.add_test_set(x_test)
    sample = gp.sample(10)
    plt.figure(2)
    plt.title('lots of train set, lots of test set')
    plt.plot(x_train, y_train, 'o')
    plt.plot(x_test, sample, 'o')

    ##########################################
    gp.refresh()
    x_train = np.arange(0, 10).reshape(-1,1)
    y_train = x_train * 2 + 2
    gp.add_train_set(x_train, y_train)

    x_train2 = np.array([[20]])
    y_train2 = np.array([[42]])
    gp.add_train_set(x_train2, y_train2)

    x_test = np.arange(10,20).reshape(-1,1)
    gp.add_test_set(x_test)
    sample = gp.sample(10)
    plt.figure(3)
    plt.title('lots of train set, lots of test set, add goal')
    plt.plot(x_train, y_train, 'o')
    plt.plot(x_train2, y_train2, 'o')
    plt.plot(x_test, sample, 'o')

    plt.show()

def test4(debug_mode = []):
    # test the gaussian process functions
    print "#################TEST4#################"
    print "# TWO D GAUSSIAN PROCESS TEST"
    print "#######################################"

    twodgp = util.TwoD_Gaussian_Process(x_kernel, y_kernel)
    twodgp.set_debug_mode(debug_mode)

    # 1 step from goal
    print "test case 1: past time is one step from goal"
    sample_count = 10
    init_position = np.array([20., 30.])
    velocity = np.array([1., 2.])
    past_time = np.arange(0, 10).reshape(-1,1)
    past_path = np.outer(past_time, velocity) + init_position
    path_history = util.Path_History(past_time, past_path)
    goal_time = 10
    goal_position = goal_time * velocity + init_position
    future_time, sample_future_path = twodgp.sample(path_history, goal_time, goal_position, sample_count)

    plt.figure(1)
    plt.title('past time is one step from goal')
    plt.axis([10, 40, 20, 60])
    plt.plot(past_path[:, 0], past_path[:, 1], 'x')
    plt.plot(goal_position[0], goal_position[1], 'x')
    plt.plot(sample_future_path[:,:,0], sample_future_path[:,:,1], 'o')


    # 1 history multiple step from goal
    print "test case 2: 1 history multiple step from goal"
    sample_count = 10
    init_position = np.array([20., 30.])
    velocity = np.array([1., 2.])
    init_time = 0
    path_history = util.Path_History(0, init_position)
    goal_time = 10
    goal_position = goal_time * velocity + init_position
    future_time, sample_future_path = twodgp.sample(path_history, goal_time, goal_position, sample_count)

    plt.figure(2)
    plt.title('one history with multiple step gap from goal')
    plt.axis([10, 40, 20, 60])
    past_path = path_history.past_path
    plt.plot(past_path[:, 0], past_path[:, 1], 'x')
    plt.plot(goal_position[0], goal_position[1], 'x')
    sample_future_path = np.vstack((np.tile(init_position, (1,sample_count, 1)), sample_future_path))
    plt.plot(sample_future_path[:,:,0], sample_future_path[:,:,1], '-')
        
    # multiple history and multiple step from goal
    print "test case 3: multiple history multiple step from goal"
    sample_count = 10
    init_position = np.array([20., 30.])
    velocity = np.array([1., 2.])
    past_time = np.arange(0, 10).reshape(-1,1)
    past_path = np.outer(past_time, velocity) + init_position
    path_history = util.Path_History(past_time, past_path)

    goal_time = 20
    goal_position = goal_time * velocity + init_position
    future_time, sample_future_path = twodgp.sample(path_history, goal_time, goal_position, sample_count)

    plt.figure(3)
    plt.title('multiple history with multiple step gap from goal')
    plt.axis([10, 50, 20, 80])
    plt.plot(past_path[:, 0], past_path[:, 1], 'x')
    plt.plot(goal_position[0], goal_position[1], 'x')
    past_path = np.expand_dims(past_path, 1)
    sample_past_path = np.tile(past_path, (1, sample_count, 1))
    sample_future_path = np.vstack((sample_past_path, sample_future_path))
    plt.plot(sample_future_path[:,:,0], sample_future_path[:,:,1], '-')

    plt.show()

def test5(debug_mode = []):
    # test the interaction function, weighted path and calculation
    print "#################TEST4#################"
    print "# INTERACTION FUNCTION TEST"
    print "#######################################"

    # one time step all same sample dist all 1, 0
    print "test case 1: same time length test case"
    print "weight calculation test"
    weight = np.ones(10) * (1-.1*np.exp(-2.))
    print "should print: ", weight
    ped1_time = np.array([10])
    ped2_time = np.array([10])
    ped1_path = np.tile(np.array([1., 2.]), (1,10,1))
    ped2_path = np.tile(np.array([2., 2.]), (1,10,1))
    interaction_parameters = {'h':.5, 'alpha':.1}
    weight = util.weight_calculation(ped1_time, ped1_path,
            ped2_time, ped2_path, interaction_parameters,
            debug_mode)
    print weight


    print "weight next position path test"
    print "should print: ", np.array([1., 2.])
    next_position = util.weighted_next_position(ped1_path, weight, debug_mode)
    print next_position

    weight[9] = 1000.
    ped1_path[0][9][0] = 3.
    ped1_path[0][9][1] = 5.
    print "should be very close to [3.,5.]"
    next_position = util.weighted_next_position(ped1_path, weight, debug_mode)
    print next_position

    print ""
    print "--------------------------------------"
    # one time step versus two time step also dist 1, 0
    print "test case 2: different time length test case"
    print "weight calculation test"
    weight = np.ones(10) * (1-.1*np.exp(-2.))
    print "should print: ", weight
    ped1_time = np.array([10])
    ped2_time = np.array([10])
    ped1_path = np.tile(np.array([1., 2.]), (1,10,1))
    ped2_path_part1 = np.tile(np.array([2., 2.]), (1,10,1))
    ped2_path_part2 = np.tile(np.array([3., 2.]), (1,10,1))
    ped2_path = np.vstack((ped2_path_part1, ped2_path_part2))
    interaction_parameters = {'h':2., 'alpha':.5}
    weight = util.weight_calculation(ped1_time, ped1_path,
            ped2_time, ped2_path, interaction_parameters,
            debug_mode)
    print weight


    print "weight path test"
    print "should print: ", np.array([1., 2.])
    next_position = util.weighted_next_position(ped1_path, weight, debug_mode)
    print next_position

    print ""
    print "--------------------------------------"
    print "test case 3: use gp to generate samples and check theirrelationship"

    # two complete path generated by twod gp three samples
    twodgp = util.TwoD_Gaussian_Process(x_kernel, y_kernel)
    twodgp.set_debug_mode(debug_mode)

    sample_count = 3
    init_position = np.array([20., 30.])
    velocity = np.array([1., 2.])
    past_time = np.arange(0, 10).reshape(-1,1)
    past_path = np.outer(past_time, velocity) + init_position
    path_history = util.Path_History(past_time, past_path)

    goal_time = 15
    goal_position = goal_time * velocity + init_position

    ped1_time, ped1_path = twodgp.sample(path_history, goal_time, goal_position, sample_count)

    #####################################
    init_position = np.array([20., 30.])
    velocity = np.array([1., 2.])
    past_time = np.arange(0, 10).reshape(-1,1)
    past_path = np.outer(past_time, velocity) + init_position
    path_history = util.Path_History(past_time, past_path)

    goal_time = 12
    goal_position = goal_time * velocity + init_position
    ped2_time, ped2_path = twodgp.sample(path_history, goal_time, goal_position, sample_count)

    weight = util.weight_calculation(ped1_time, ped1_path,
            ped2_time, ped2_path, interaction_parameters,
            debug_mode)
    print "the weight of three sample pairs: red, blue, green"
    print weight

    plt.figure(1)
    plt.axis([25,37,45,62])
    plt.plot(ped1_path[:,0,0], ped1_path[:,0,1], 'ro-')
    plt.plot(ped2_path[:,0,0], ped2_path[:,0,1], 'rx-')

    plt.plot(ped1_path[:,1,0], ped1_path[:,1,1], 'bo-')
    plt.plot(ped2_path[:,1,0], ped2_path[:,1,1], 'bx-')
    
    plt.plot(ped1_path[:,2,0], ped1_path[:,2,1], 'go-')
    plt.plot(ped2_path[:,2,0], ped2_path[:,2,1], 'gx-')

    next_position = util.weighted_next_position(ped1_path, weight, debug_mode)
    print "next position of the weighted longer path"
    plt.plot(next_position[0], next_position[1], 'y^')

    plt.show()








#test1()
#test2()
#test3([1])
test4([2])
#test5([3])
# debug mode = 1: assertion in gaussian process class
# debug mode = 2: assertion in two dimension gaussian process class



