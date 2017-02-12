import pygame
import numpy as np
import random
import time
from navigation_map import NavigationMap

class Simulation_Environment:
    """
    The simple simulation environment containing only pedestrians
    If want to include any other objects, please write another class
    based on this class
    """
    def __init__(self, screen_size, screen_name, frame_per_second):
        """
        Input:
            screen_size: (int, int); the size of screen
            screen_name: string; the name of the screen
            frame_per_second: float; diplayment frame per second
        """
        self.screen_size = screen_size
        self.screen_color = (255, 255, 255)
        self.screen_name = screen_name
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption(screen_name)
        self.screen.fill(self.screen_color)

        self.clock = pygame.time.Clock()
        self.pedestrian_list = []
        self.time_step = 1. / frame_per_second
        self.frame_per_second = frame_per_second

        self.pedestrian_relative_position = np.array([[]])
        self.all_pedestrian_velocity = np.array([[]])
        self.pedestrian_ID2index = {}
        self.debug_mode = 0
        # debug mode: 
        # 0 --> display nothing
        # 1 --> display cross pedestrian value
        # 2 --> display energy calculation detail
        # 3 --> display minimize energy process

        self.nmap = None
        self.map_restore_path = None
        self.patch_size = None
        self.energy_map = None
        self.background_map = None


    def add_navigation_map(self, patch_size, map_restore_path):
        # add navigation map to the simulation environment
        self.patch_size = patch_size
        self.map_restore_path = map_restore_path
        nmap_config = {
                'width': self.screen_size[0],
                'height': self.screen_size[1],
                'patch_size': self.patch_size
                }
        self.nmap = NavigationMap(nmap_config)
        self.nmap.restore(map_restore_path)
        self.nmap.edit()
        self.nmap.create_energy_map()
        image = self.nmap.get_map_image()
        self.background_map = pygame.image.frombuffer(image.tostring(),
                image.shape[1::-1], "RGB")
        self.energy_map = self.nmap._energy_map.T
        #self.nmap.visualize_energy_map()


    def set_debug_mode(self, debug_mode):
        # change the debug mode
        self.debug_mode = debug_mode

    def clock_tick(self):
        # make clock tick a time step
        self.clock.tick(self.frame_per_second)

    def reset_screen(self):
        # reset the screen to original format
        self.screen.fill(self.screen_color)

    def add_pedestrian(self, pedestrian):
        # add a pedestrian to the environment
        pedestrian_ID = pedestrian.ID
        pedestrian_index = len(self.pedestrian_list)
        self.pedestrian_ID2index[pedestrian_ID] = pedestrian_index
        self.pedestrian_list.append(pedestrian)
        

    def remove_pedestrian(self, pedestrian_ID_list):
        # remove a list of pedestrians from the environment by their IDs
        for pedestrian_ID in pedestrian_ID_list:
            pedestrian_index = self.pedestrian_ID2index[pedestrian_ID]
            del self.pedestrian_list[pedestrian_index]
            del self.pedestrian_ID2index[pedestrian_ID]
            for ID, index in self.pedestrian_ID2index.items():
                if index > pedestrian_index:
                    self.pedestrian_ID2index[ID] = index - 1

    def calculate_cross_pedestrian_value(self):
        # calculate cross pedestrian value for move to use
        # pedestrian_relative_position[i][j] = p[i] - p[j]
        # all_pedestrian_velocity[i] = v[i]
        pedestrian_count = len(self.pedestrian_list)
        
        if pedestrian_count != 0:
            position = self.pedestrian_list[0].position
            position_matrix = np.tile(position, (1, pedestrian_count, 1))
            for pedestrian_index in range(1, pedestrian_count):
                position = self.pedestrian_list[pedestrian_index].position
                repeat_matrix = np.tile(position, (1, pedestrian_count, 1))
                position_matrix = np.concatenate((position_matrix, repeat_matrix))
            self.pedestrian_relative_position = position_matrix - np.swapaxes(position_matrix, 0, 1) 
        else:
            self.pedestrian_relative_position = np.array([[]])

        if pedestrian_count != 0:
            self.all_pedestrian_velocity = self.pedestrian_list[0].velocity[np.newaxis, :]
            for pedestrian_index in range(1, pedestrian_count):
                velocity = self.pedestrian_list[pedestrian_index].velocity
                self.all_pedestrian_velocity = np.vstack((self.all_pedestrian_velocity, velocity))
        else:
            self.all_pedestrian_velocity = np.array([[]])

        if self.debug_mode == 1:
            print "pedestrian count: ", len(self.pedestrian_list)
            print "pedestrian ID to index:"
            for key, value in self.pedestrian_ID2index.items():
                print "     ID: ", key, "; index: ", value
            print "pedestrian relative position: [i][j] = p[i] - p[j]: "
            print "    ", self.pedestrian_relative_position
            print "all pedestrian velocity: \n    ", self.all_pedestrian_velocity
            print ""


    def display(self):
        # display all objects on the screen without flipping
        if self.background_map is not None:
            self.screen.blit(self.background_map, [0,0])

        for pedestrian in self.pedestrian_list:
            pedestrian.display()

    def move(self):
        # move all objects to the next step it should be virtually
        for pedestrian in self.pedestrian_list:
            pedestrian.move()

    def run(self):
        # simply run the simulation
        running = True
        while running:
            #self.screen.fill(self.screen_color)
            self.clock.tick(self.frame_per_second)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.calculate_cross_pedestrian_value()
            self.move()
            self.display()
            pygame.display.flip()


class Pedestrian:
    def __init__(self, parameters, simulation_environment):
        """
        Pedestrian represent the simulated pedestrian in the simulation environment
        Input:
            parameters: dictionary; all the useful parameters while setup the pedestrian
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
            simulation_environment: Simulation_Environment; the simulation environment
                                    that contains this pedestrian
        """
        self.simulation_environment = simulation_environment
        self.screen = simulation_environment.screen
        self.frame_per_second = simulation_environment.frame_per_second
        self.time_step = simulation_environment.time_step
        self.debug_mode = simulation_environment.debug_mode

        self.ID = parameters['ID']
        self.lambda1 = parameters['lambda1']
        self.lambda2 = parameters['lambda2']
        self.sigma_d = parameters['sigma_d']
        self.sigma_w = parameters['sigma_w']
        self.beta = parameters['beta']
        self.alpha = parameters['alpha']
        self.pixel2meters = parameters['pixel2meters']
        self.meter2pixels = 1. / self.pixel2meters
        
        self.expected_speed = parameters['expected_speed']
        self.goal_position = parameters['goal_position']
        self.velocity = parameters['initial_velocity']
        self.position = parameters['initial_position']
        

    def _minimize_energy_velocity(self):
        """
        calculate the velocity that minimize the energy by RMSprop
        Return:
            velocity: np.1darray; the velocity that minimize the energy, in pixel/s
        """
        time_start = time.time()
        parameters = {}
        parameters['gamma'] = 0.99
        parameters['alpha'] = 0.5 #0.001
        parameters['epsilon'] = 10**(-4)

        initial_velocity = self.velocity
        energy_list, minimize_energy_velocity = RMSprop(
                initial_velocity, self._energy_with_gradient, parameters)
        time_end = time.time()

        if self.debug_mode == 3:
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
        sim_env = self.simulation_environment
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
        index = sim_env.pedestrian_ID2index[ID]
        ped_count = len(sim_env.pedestrian_list)

        gvi2pixel = np.array([0., 0.])
        E_i = 0.

        if ped_count != 1:
            # if there is more than one pedestrian, calculate social energy
            k = np.delete(sim_env.pedestrian_relative_position[index], index, axis = 0) * p2m # relative position 
            q = np.tile(v, (ped_count - 1, 1)) - np.delete(sim_env.all_pedestrian_velocity, index, axis = 0) * p2m

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

            if self.debug_mode == 2:
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
            if self.debug_mode == 2:
                print "##########current pedestrian index: ", index
                print "Speed energy S: ", E_s
                print "direction energy D: ", E_d
                print "social energy I:  0."
                print "total energy E: ", E_s + E_d

        # sum energy and energy gradient together
        energy = E_s + E_d + E_i 
        energy_gradient = gvs2pixel + gvd2pixel + gvi2pixel

        if self.simulation_environment.nmap is not None:
            # there is navigation map, add environment energy
            energy_map = self.simulation_environment.energy_map
            screen_size = self.simulation_environment.screen_size
            map_x = int(self.position[0])
            map_y = int(self.position[1])
            # calculate gradient by difference between left, right, up, down 2 pixels
            gradient_gap = 7
            if map_x >= 0 and map_x < screen_size[0] and map_y >= 0 and map_y < screen_size[1]:
                energy += energy_map[map_x][map_y]

                left_x = max(0, map_x - gradient_gap)
                right_x = min(screen_size[0] - 1, map_x + gradient_gap)
                up_y = max(0, map_y - gradient_gap)
                down_y = min(screen_size[1] - 1, map_y + gradient_gap)

                dedx = (energy_map[right_x][map_y] - energy_map[left_x][map_y]) / (right_x - left_x)
                dedy = (energy_map[map_x][down_y] - energy_map[map_x][up_y]) / (down_y - up_y)
                dedvx = dedx
                dedvy = dedy

                energy_gradient += np.array([dedvx, dedvy])
        return (energy, energy_gradient)

    def move(self):
        # move the pedestrian to next position
        optimal_velocity = self._minimize_energy_velocity()
        self.position += (self.alpha * self.velocity + (1 - self.alpha) * 
                optimal_velocity) * self.time_step
        self.velocity = optimal_velocity
        return

    def display(self):
        # display the pedestrian on the screen before flipping
        pygame.draw.circle(self.screen, (0,0,0), (int(self.position[0]), int(self.position[1])), 3, 0)
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

