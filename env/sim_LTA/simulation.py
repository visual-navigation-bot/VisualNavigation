import numpy as np
import random
import time

class LTA:
    """
    LTA Simulation Class 
    inherit by Simulation Class
    """
    def __init__(self, field_size, frame_per_second):
        """
        Input:
            field_size: (float, float); the size of field;
            frame_per_second: float; diplayment frame per second
        """
        self.field_size = field_size

        self.ped_list = []
        self.agent = None
        self.time_step = 1. / frame_per_second
        self.frame_per_second = frame_per_second

        self.ped_relative_position = np.array([], dtype = np.float32).reshape((0,0,2))
        self.all_ped_velocity = np.array([], dtype = np.float32).reshape((0,2))
        self.ped_ID = np.array([], dtype = np.int8).reshape(0)
        self.debug_mode = []
        # debug mode: 
        # 1 --> display cross pedestrian value
        # 2 --> display energy calculation detail
        # 3 --> display minimize energy process

    def add_ped(self, ped_params):
        """
        Add a pedestrian to the Simulation
        Input:
            ped_params: dictionary; all the useful parameters while setup the pedestrian
                ID: int; the ID of this pedestrian
                lambda1: float; lambda1 in energy equation
                lambda2: float; lambda2 in energy equation
                sigma_d: float; sigma_d in energy equation
                sigma_w: float; sigma_w in energy equation
                beta: float; beta in energy equation
                alpha: float; alpha to update position
                pixel2meters: float; 1 pixel on screen represents this much meters in real world
                expected_speed: float; the expected speed in pixel/s
                goal_position: np.1darray; the goal position in pixel
                initial_velocity: np.1darray; the initial velocity in pixel/s
                initial_position: np.1darray; the initial position in pixel
        """
        ped = Pedestrian(ped_params, self)
        ped_ID = ped.ID
        self.ped_ID = np.hstack((self.ped_ID, ped_ID))
        self.ped_list.append(ped)

    def move(self):
        """
        move all pedestrian except agent for one time step
        remove any pedestrian that is out of the field or arrive destination
        """
        for ped in self.ped_list:
            if ped.ID == -1:
                # agent
                if self.agent.is_done():
                    self.remove_pedestrian(-1)
                    print "Agent removed!!!!!"
            else:
                if np.linalg.norm(ped.position - ped.goal_position) < 20:
                    self._remove_pedestrian(ped.ID)
        self._calculate_cross_pedestrian_value() # speed up by numpy
        for ped in self.ped_list:
            if ped.ID != -1:
                ped.move()

    def get_ped_state(self):
        """
        Return the state and ID of the pedestrians, don't return agent's information
        Input:
            None
        Return:
            ped_state: dictionary;
                ped_ID: numpy1darray int8; the pedestrian ID
                ped_position: numpy2darray float32; the pedestrian position
                ped_velocity: numpy2darray float32; the pedestrian velocity
        """
        ped_state = {}
        ped_position = np.array([], dtype = np.float32).reshape((0,2))
        ped_velocity = np.array([], dtype = np.float32).reshape((0,2))
        ped_ID = np.array([], dtype = np.int8).reshape(0)
        for ped in self.ped_list:
            if ped.ID != -1:
                # if it is not agent
                ped_velocity = np.vstack((ped_velocity, ped.velocity))
                ped_position = np.vstack((ped_position, ped.position))
                ped_ID = np.hstack((ped_ID, ped.ID))
        ped_state['ped_ID'] = ped_ID
        ped_state['ped_position'] = ped_position
        ped_state['ped_velocity'] = ped_velocity
        return ped_state

    def add_agent(self, agent):
        """
        Add the agent in the Simulation
        Input:
            agent: Agent (defined by environment class); the agent
        Note:
            agent class have propertis: ID = -1, velocity, position
        """
        self.agent = agent
        assert agent.ID == -1, "Error: agent's ID is not -1"
        self.ped_ID = np.hstack((self.ped_ID, agent.ID))
        self.ped_list.append(agent)



    def get_ped_count(self):
        """
        Return the numbers of pedestrians
        """
        if self.agent is not None:
            return len(self.ped_list) - 1
        else:
            return len(self.ped_list)

    def set_debug_mode(self, debug_mode = []):
        # change the debug mode
        self.debug_mode = debug_mode

    def _remove_pedestrian(self, ped_ID):
        # remove a list of pedestrians from the environment by their IDs
        ped_index = np.argwhere(self.ped_ID == ped_ID)[0][0]
        del self.ped_list[ped_index]
        self.ped_ID = np.delete(self.ped_ID, ped_index, 0)

    def _calculate_cross_pedestrian_value(self):
        # calculate cross pedestrian value for move to use
        # pedestrian_relative_position[i][j] = p[i] - p[j]
        # all_pedestrian_velocity[i] = v[i]
        ped_count = len(self.ped_list)


        position_matrix = np.array([], dtype = np.float32).reshape((0,ped_count,2))
        for ped_index in range(ped_count):
            position = self.ped_list[ped_index].position
            repeat_matrix = np.tile(position, (1, ped_count, 1))
            position_matrix = np.concatenate((position_matrix, repeat_matrix))
        self.ped_relative_position = position_matrix - np.swapaxes(position_matrix, 0, 1) 

        self.all_ped_velocity = np.array([], dtype = np.float32).reshape((0,2))
        for ped_index in range(ped_count):
            velocity = self.ped_list[ped_index].velocity
            self.all_ped_velocity = np.vstack((self.all_ped_velocity, velocity))

        if 1 in self.debug_mode:
            print "pedestrian count: ", len(self.ped_list)
            print "pedestrian ID to index:"
            for ped_index in range(len(self.ped_list)):
                ped_ID = self.ped_ID[ped_index]
                print "     ID: ", ped_ID, "; index: ", ped_index

            print "pedestrian relative position: [i][j] = p[i] - p[j]: "
            print "    ", self.ped_relative_position
            print "all pedestrian velocity: \n    ", self.all_ped_velocity
            print ""

class Pedestrian:
    def __init__(self, params, sim):
        """
        Pedestrian represent the simulated pedestrian in the simulation environment
        Input:
            params: dictionary; all the useful parameters while setup the pedestrian
                ID: int; the ID of this pedestrian
                lambda1: float; lambda1 in energy equation
                lambda2: float; lambda2 in energy equation
                sigma_d: float; sigma_d in energy equation
                sigma_w: float; sigma_w in energy equation
                beta: float; beta in energy equation
                alpha: float; alpha to update position
                pixel2meters: float; 1 pixel on screen represents this much meters in real world
                expected_speed: float; the expected speed in pixel/s
                goal_position: np.1darray; the goal position in pixel
                initial_velocity: np.1darray; the initial velocity in pixel/s
                initial_position: np.1darray; the initial position in pixel
            sim: Simulation; the simulation environment
                                    that contains this pedestrian
        """
        self.sim = sim
        self.frame_per_second = sim.frame_per_second
        self.time_step = sim.time_step
        self.debug_mode = sim.debug_mode

        self.ID = params['ID']
        self.lambda1 = params['lambda1']
        self.lambda2 = params['lambda2']
        self.sigma_d = params['sigma_d']
        self.sigma_w = params['sigma_w']
        self.beta = params['beta']
        self.alpha = params['alpha']
        self.pixel2meters = params['pixel2meters']
        self.meter2pixels = 1. / self.pixel2meters
        
        self.expected_speed = params['expected_speed']
        self.goal_position = params['goal_position']
        self.velocity = params['initial_velocity']
        self.position = params['initial_position']
        

    def _minimize_energy_velocity(self):
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

        initial_velocity = self.velocity
        energy_list, minimize_energy_velocity = RMSprop(
                initial_velocity, self._energy_with_gradient, params)
        time_end = time.time()

        if 3 in self.debug_mode:
            print "energy decay process in RMSprop: "
            for energy in energy_list:
                print energy
            print "velocity that minimize energy: ", minimize_energy_velocity
            print "steps:", len(energy_list)
            print "process time: ", time_end - time_start

        return minimize_energy_velocity

    def _energy_with_gradient(self, velocity):
        """
        A function that calculates the energy and gradient of energy
        while given the velocity, the velocity is given
        in pixel unit, energy is in meters unit,
        and gradient of energy is in pixel unit
        Input:
            velocity: np.2darray; the velocity this pedestrian taken
        Return:
            energy: float; given this velocity, the energy for this pedestrian
            energy_gradient: np.2darray; the gradient of energy to the velocity
        """
        sim = self.sim
        v = self.pixel2meters * velocity
        vt = self.pixel2meters * self.velocity #last time step velocity
        u = self.pixel2meters * self.expected_speed
        p2m = self.pixel2meters
        l1 = self.lambda1
        l2 = self.lambda2
        sd = self.sigma_d
        sw = self.sigma_w
        b = self.beta
        z = self.pixel2meters * self.goal_position
        p = self.pixel2meters * self.position

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
        ID = self.ID
        index = np.argwhere(sim.ped_ID == ID)[0][0]
        ped_count = len(sim.ped_list)

        gvi2pixel = np.array([0., 0.])
        E_i = 0.

        if ped_count != 1:
            # if there is more than one pedestrian, calculate social energy
            k = np.delete(sim.ped_relative_position[index], index, axis = 0) * p2m # relative position 
            q = np.tile(v, (ped_count - 1, 1)) - np.delete(sim.all_ped_velocity, index, axis = 0) * p2m

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

            if 2 in self.debug_mode:
                print "##########current pedestrian index: ", index
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
            if 2 in self.debug_mode:
                print "##########current pedestrian index: ", index
                print "Speed energy S: ", E_s
                print "direction energy D: ", E_d
                print "social energy I:  0."
                print "total energy E: ", E_s + E_d

        # sum energy and energy gradient together
        energy = E_s + E_d + E_i 
        energy_gradient = gvs2pixel + gvd2pixel + gvi2pixel
        return (energy, energy_gradient)

    def move(self):
        # move the pedestrian to next position
        optimal_velocity = self._minimize_energy_velocity()
        self.position += (self.alpha * self.velocity + (1 - self.alpha) * 
                optimal_velocity) * self.time_step
        self.velocity = optimal_velocity
        return

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

