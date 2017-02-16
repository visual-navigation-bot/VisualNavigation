import numpy as np
import random
import time

def LTA_Controller(observations):
    """
    LTA Controller Function
    Designed for LTA Continuous Ver0 environments
    Input:
        observation: dictionary;
            agent_ID: -1
            agent_position: np.1darray; the position of agent
            agent_velocity: np.1darray; the velocity of agent
            agent_goal_position: np.1darray; the goal position of agent
            agent_expected_speed: float; the expected speed of agent
            ped_ID: np.1darray int8; ID of other pedestrians
            ped_position: np.2darray float32; axis 0 is agent index, axis 1 is agent position
            ped_velocity: np.2darray float32; axis 0 is agent index, axis 1 is agent velocity
    Return:
        action: np1darray; acceleration
    """
    position = observations['agent_position']
    velocity = observations['agent_velocity']
    goal_position = observations['agent_goal_position']
    expected_speed = observations['agent_expected_speed']

    ped_ID = observations['ped_ID']
    ped_position = observations['ped_position']
    ped_velocity = observations['ped_velocity']

    lambda1 = 2.33
    lambda2 = 2.073
    sigma_d = 0.361
    sigma_w = 2.088
    beta = 1.462
    step_time = 0.4
    p2m = 0.02

    debug_mode = False

    def _minimize_energy_velocity():
        """
        calculate the velocity that minimize the energy by RMSprop
        Return:
            velocity: np.1darray; the velocity that minimize the energy, in pixel/s
        """
        time_start = time.time()
        params = {}
        params['gamma'] = 0.99
        params['alpha'] = 0.5 #0.001
        params['epsilon'] = 10**(-4)

        initial_velocity = velocity.copy() + np.random.random((2)) * 2.5
        energy_list, minimize_energy_velocity = RMSprop(
                initial_velocity, _energy_with_gradient, params)
        time_end = time.time()

        if debug_mode:
            print "energy decay process in RMSprop: "
            for energy in energy_list:
                print energy
            print "velocity that minimize energy: ", minimize_energy_velocity
            print "steps:", len(energy_list)
            print "process time: ", time_end - time_start

        return minimize_energy_velocity

    def _energy_with_gradient(next_v):
        """
        A function that calculates the energy and gradient of energy
        while given the velocity, the velocity is given
        in pixel unit, energy is in meters unit,
        and gradient of energy is in pixel unit
        Input:
            next_v: np.2darray; the velocity this pedestrian taken
        Return:
            energy: float; given this velocity, the energy for this pedestrian
            energy_gradient: np.2darray; the gradient of energy to the velocity
        """
        v = p2m * next_v
        vt = p2m * velocity
        u = p2m * expected_speed
        l1 = lambda1
        l2 = lambda2
        sd = sigma_d
        sw = sigma_w
        b = beta
        z = p2m * goal_position
        p = p2m * position

        # using back propogation in the following
        # E_s (speed) = lambda1 * (u - |v|)**2
        normv = np.linalg.norm(v) 
        E_s = l1 * (u - normv) ** 2

        gnormv = - 2 * l1 * (u - normv)
        gvs = gnormv * v / normv
        gvs2pixel = gvs * p2m

        # E_d (direction) = - (p dot v) / (|p| * |v|)
        pdotv = np.dot((z - p), v)
        normv = np.linalg.norm(v)
        normpnormv = np.linalg.norm((z-p)) * normv
        E_d = -l2 * pdotv / normpnormv

        gpdotv = -l2 / normpnormv
        gnormpnormv = l2 * pdotv / normpnormv ** 2
        gnormv = gnormpnormv * np.linalg.norm(z-p)
        gvd = gnormv * v / normv
        gvd += gpdotv * (z - p)
        gvd2pixel = gvd * p2m

        # E_i = sigma(i)(wr(i) * exp(- d**2 / (2**sd**2)))
        # q = v - vj; k = pi - pj; cos(phi) = -kdotvt / (|k|*|vt|)
        # i is this pedestrian
        # d = k - kdotq * q / |q|**2
        # wr = exp(-k ** 2 / (2 * sw**2)) * ( (1+cos(phi)) / 2)**beta
        ped_count = len(ped_ID)

        gvi2pixel = np.array([0., 0.])
        E_i = 0.

        if ped_count != 0:
            # if there is more than one pedestrian (including agent), calculate social energy
            # agent's position subtracted by other pedestrian's position
            # agent's velocity subtracted by other pedestrian's velocity
            k = (position - ped_position) * p2m
            q = (velocity - ped_velocity) * p2m

            kdotq = np.sum(k * q, axis = 1) 
            normq = np.linalg.norm(q, axis = 1) 
            t = - kdotq / normq ** 2 #kdotq / |q|**2
            mask = t>0
            maskt = mask * t
            d = k + q * maskt[:, np.newaxis]
            normd = np.linalg.norm(d, axis = 1)
            E_v = np.exp( - normd**2 / (2 * sd**2))
            wd = np.exp(- np.linalg.norm(k, axis = 1)**2 / (2 * sw**2))
            cos = - np.dot(vt, np.swapaxes(k, 0, 1)) / (np.linalg.norm(vt) * np.linalg.norm(k, axis = 1))
            wphi = ((1 + cos) / 2)**b
            E_i = np.sum(wphi * wd * E_v)


            gE_v = wphi * wd
            gnormd = gE_v * E_v * (- normd / sd**2)
            gd = (gnormd / normd)[:, np.newaxis] * d
            gmaskt = np.sum(q * gd, axis = 1)
            gq = gd * maskt[:, np.newaxis]
            gt = gmaskt * mask
            gnormq = 2 *gt * kdotq / normq**3
            gq += (gnormq / normq)[:, np.newaxis] * q
            gkdotq = - gt / normq**2
            gq += gkdotq[:, np.newaxis] * k
            gvi = np.sum(gq, axis = 0)
            gvi2pixel = gvi * p2m

            if debug_mode:
                print "wd: ", wd
                print "wphi: ", wphi
                print "k: ", k
                print "q: ", q
                print "d: ", d
                print "E: ", E_v
                print "Speed energy S: ", E_s
                print "direction energy D: ", E_d
                print "social energy I: ", E_i
                print "total energy E: ", E_i + E_s + E_d
                print ""
                print "gI: ", gvi2pixel
                print "gS: ", gvs2pixel
                print "gD: ", gvd2pixel
        else:
            if debug_mode:
                print "##########"
                print "Speed energy S: ", E_s
                print "direction energy D: ", E_d
                print "social energy I:  0."
                print "total energy E: ", E_s + E_d

        # sum energy and energy gradient together
        energy = E_s + E_d + E_i 
        energy_gradient = gvs2pixel + gvd2pixel + gvi2pixel
        return (energy, energy_gradient)

    action = (_minimize_energy_velocity() - velocity) / step_time

    return action



def RMSprop(initial_value, energy_and_gradient_function, parameters):
    """
    Use RMSprop gradient descent to find the minimum of a function
    Input:
        initial_value: ?; the initial value to calculate the minimum by gradient descent
        energy_and_gradient_function: function; a function to calculate energy and its gradient given value 
                                      (same type as initial value)
        parameters: dictionary; a dictionary for all used parameters in RMSprop
            gamma, alpha (learning rate), epsilon: when to stop updating
    Return:
        energy_list: list of float; the process of gradient descent, shown by energy
        minimum_value: ?; the value that minimize the energy function
    """
    gamma = parameters['gamma']
    alpha = parameters['alpha']
    epsilon = parameters['epsilon']

    value = initial_value
    value_list = [initial_value]
    energy, energy_gradient = energy_and_gradient_function(value)
    energy_list = [energy]
    running_average = 0.0001

    done = False
    while not done:
        running_average = (1 - gamma) * np.square(energy_gradient) + running_average
        delta_value = alpha * energy_gradient / np.sqrt(running_average)
        value = value - delta_value
        value_list.append(value)
        updated_energy, energy_gradient = energy_and_gradient_function(value)
        delta_energy = abs(energy - updated_energy)
        if delta_energy < 10**-12 or abs(delta_energy / energy) < epsilon:
            done = True
        energy = updated_energy
        energy_list.append(energy)
    return (energy_list, value)

