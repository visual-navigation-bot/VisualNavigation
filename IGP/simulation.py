import pygame
import numpy as np
import time
import util
import itertools

class Simulation_Environment:
    def __init__(self, screen_size, screen_name, frame_per_second):
        self.screen_size = screen_size
        self.screen_name = screen_name
        self.screen_color = (255,255,255)
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption(screen_name)
        self.screen.fill(self.screen_color)

        self.frame_per_second = frame_per_second
        self.time_step = 1. / frame_per_second
        self.clock = pygame.time.Clock()
        self.debug_mode = []

        self.pedestrian_list = []
        self.ID2index = {}
        self.current_time = 0

    def set_debug_mode(self, debug_mode = []):
        # change the debug mode
        self.debug_mode = debug_mode
        return

    def flip(self):
        # call pygame display flip
        pygame.display.flip()

    def clock_tick(self):
        # make clock tick a time step
        self.clock.tick(self.frame_per_second)
        return

    def reset_screen(self):
        # reset the screen by re fill the color
        self.screen.fill(self.screen_color)
        return

    def add_pedestrian(self, pedestrian):
        # add the pedestrian to the pedestrian list
        self.pedestrian_list.append(pedestrian)
        pedestrian_ID = pedestrian.ID
        pedestrian_index = len(self.pedestrian_list) - 1
        self.ID2index[pedestrian_ID] = pedestrian_index
        return

    def remove_pedestrian(self, pedestrian_ID_list):
        # remove the pedestrian from the pedestrian list by IDs
        for pedestrian_ID in pedestrian_ID_list:
            pedestrian_index = self.ID2index[pedestrian_ID]
            del self.pedestrian_list[pedestrian_index]

            # change ID2index, those index larger than removed one will descrease 1
            for ID, index in self.ID2index.items():
                if index > pedestrian_index:
                    self.ID2index[ID] = index - 1
        return
    
    def find_next_position(self):
        # update all the pedestrians location by IGP
        for pedestrian in self.pedestrian_list:
            pedestrian.find_next_position()
        return

    def update(self):
        # update all the pedestrians location by IGP
        self.current_time = self.current_time + 1
        if 4 in self.debug_mode:
            print "==========ANOTHER TIME STEP========"
            print "current time : ", self.current_time
        for pedestrian in self.pedestrian_list:
            pedestrian.update()
        return

    def display(self):
        # display all the pedestrians on the screen before flipping
        for pedestrian in self.pedestrian_list:
            pedestrian.display()

    def run(self):
        # run the simulation in the smplest case
        running = True
        while running:
            self.clock_tick()
            self.reset_screen()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.find_next_position()
            self.update()
            self.display()
            pygame.display.flip()
        return

class Pedestrian:
    def __init__(self, parameters, simulation_environment):
        self.simulation_environment = simulation_environment
        self.frame_per_second = simulation_environment.frame_per_second
        self.time_step = simulation_environment.time_step
        self.screen = simulation_environment.screen
        self.debug_mode = simulation_environment.debug_mode
        self.terminated = False

        self.ID = parameters['ID']
        self.position = parameters['initial_position']
        self.goal_position = parameters['goal_position']
        self.expected_speed = parameters['expected_speed']
        self.sample_count = parameters['sample_count']
        self.x_kernel = parameters['x_kernel']
        self.y_kernel = parameters['y_kernel']
        # two parameters for interaction
        self.h = parameters['h']
        self.alpha = parameters['alpha']

        self.current_time = self.simulation_environment.current_time
        self.start_time = self.current_time
        to_goal_time = int(np.linalg.norm(self.goal_position - self.position) / (self.expected_speed * self.time_step))
        if to_goal_time == 0:
            self.terminated = True
        self.goal_time = to_goal_time + self.current_time
        self.twod_gp_model = util.TwoD_Gaussian_Process(self.x_kernel, self.y_kernel)
        self.path_history = util.Path_History(self.current_time, self.position)
        self.interaction_parameters = {'alpha': self.alpha, 'h':self.h}

        self.next_position = None

    def find_next_position(self):
        # use interacting gaussian process to evaluate next position
        ped_count = len(self.simulation_environment.pedestrian_list)

        self.current_time = self.simulation_environment.current_time
        if self.terminated:
            self.next_position = self.goal_position.copy()
            return
        # sampling the path of this pedestrian
        self.future_time, self.sample_future_path = self.twod_gp_model.sample(
                self.path_history, self.goal_time, 
                self.goal_position, self.sample_count)
        self.sample_weight = np.ones(self.sample_count)

        # sampling the path of other pedestrians
        ped_sample_future_path_list = []
        ped_future_time_list = []
        self.index = self.simulation_environment.ID2index[self.ID] # get my index
        for ped_index in range(ped_count):
            if ped_index == self.index:
                continue
            ped = self.simulation_environment.pedestrian_list[ped_index]
            if ped.terminated:
                continue

            ped_goal_position = ped.goal_position
            ped_goal_time = ped.goal_time
            ped_path_history = ped.path_history
            # the seen history for the main ped
            sliced_history = ped_path_history.sliced_history(self.start_time)

            ped_future_time, ped_sample_future_path = self.twod_gp_model.sample(sliced_history, 
                    ped_goal_time, ped_goal_position, self.sample_count)
            ped_sample_future_path_list.append(ped_sample_future_path)
            ped_future_time_list.append(ped_future_time)

        ped_future_time_list.append(self.future_time)
        ped_sample_future_path_list.append(self.sample_future_path)
        working_ped_count = len(ped_sample_future_path_list)
        ped_index_pair = itertools.permutations(range(working_ped_count), 2)
        for (ped1_index, ped2_index) in ped_index_pair:
            ped1_future_time = ped_future_time_list[ped1_index]
            ped2_future_time = ped_future_time_list[ped2_index]
            ped1_sample_future_path = ped_sample_future_path_list[ped1_index]
            ped2_sample_future_path = ped_sample_future_path_list[ped2_index]
            ped_weight = util.weight_calculation(
                    ped1_future_time, ped1_sample_future_path,
                    ped2_future_time, ped2_sample_future_path,
                    self.interaction_parameters, self.debug_mode)
            self.sample_weight *= ped_weight

        next_position = util.weighted_next_position(self.sample_future_path, 
                self.sample_weight, self.debug_mode)
        self.next_position = next_position
        return

    def update(self):
        """
        Always have to call find next position before update, find next position 
        just find next position, update actually add history and update position
        and also termination of the history
        """
        self.current_time = self.simulation_environment.current_time
        if self.current_time == self.goal_time:
            self.terminated = True
        self.position = self.next_position.copy()
        self.path_history.add_history(self.current_time, self.position)
        if 4 in self.debug_mode:
            print "---------------------------------------"
            print "pedestrian ID: ", self.ID
            print "position: ", self.position
            print "termination status: ", self.terminated
            self.path_history.pprint()

        return

    def display(self):
        # display the pedestrian on the screen before flipping
        display_position = (int(self.position[0]), int(self.position[1]))
        pygame.draw.circle(self.screen, (0,0,0), display_position, 3, 0)
        return

