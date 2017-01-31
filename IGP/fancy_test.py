import pygame
import simulation
import numpy as np
import random
import util
import time

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

class Multiple_Destination:
    def __init__(self, debug_mode, parameters):
        self.screen_size = parameters['screen_size']
        self.screen_name = parameters['screen_name']
        self.frame_per_second = parameters['frame_per_second']
        self.destination_count = parameters['destination_count']
        self.max_pedestrian_count = parameters['max_pedestrian_count']
        self.sample_count = parameters['sample_count']
        self.rolling = parameters['rolling']

        self.min_destination_distance = 50.
        self.destination_list = []

        self.sim_env = simulation.Simulation_Environment(self.screen_size, 
                self.screen_name, self.frame_per_second)
        self.sim_env.set_debug_mode(debug_mode)
        self.screen = self.sim_env.screen
        self.next_ID = 0

        # generate pedestrian destinations
        while len(self.destination_list) < self.destination_count:
            new_destination = np.array([random.uniform(0, self.screen_size[0]), 
                random.uniform(0, self.screen_size[1])])
            allowed = True
            for destination in self.destination_list:
                if np.linalg.norm(new_destination - destination) < self.min_destination_distance:
                    allowed = False
            if allowed:
                self.destination_list.append(new_destination)
    
        # generate pedestrians
        for ped_index in range(self.max_pedestrian_count):
            self.add_pedestrian()

    def add_pedestrian(self):
        # add a new pedestrian to the simulation environment
        expected_speed = random.uniform(40., 80.)
        destination = self.destination_list[random.randint(1, self.destination_count) - 1]
        source = random.choice(['up','down', 'left','right'])
        initial_position = np.array([random.uniform(0, self.screen_size[0]), 
            random.uniform(0, self.screen_size[1])])
        if source == 'up':
            initial_position[1] = random.uniform(-100, 0)
        if source == 'down':
            initial_position[1] = random.uniform(0, 100) + self.screen_size[1]
        if source == 'left':
            initial_position[0] = random.uniform(-100, 0)
        if source == 'right':
            initial_position[0] = random.uniform(0, 100) + self.screen_size[0]

        sample_count = self.sample_count
        h = random.uniform(15,25)
        alpha = 0.999

        print "ID: ", self.next_ID
        print "expected speed: ", expected_speed
        print "goal position: ", destination
        print "sample count: ", sample_count
        print "initial position: ", initial_position
        print ""

        parameters = {
                'ID' : self.next_ID,
                'initial_position': initial_position,
                'goal_position': destination,
                'expected_speed': expected_speed,
                'sample_count': sample_count,
                'x_kernel': x_kernel,
                'y_kernel': y_kernel,
                'h': h,
                'alpha': alpha
                    }
        ped = simulation.Pedestrian(parameters, self.sim_env)
        self.sim_env.add_pedestrian(ped)
        self.next_ID += 1

    def display_destination(self):
        # display destinations
        for destination in self.destination_list:
            position = (int(destination[0]), int(destination[1]))
            pygame.draw.circle(self.screen, (255,0,0), position, 7, 0)

    def run(self):
        running = True
        while running:
            self.sim_env.reset_screen()
            self.sim_env.clock_tick()

            removed_ID = []
            for ped in self.sim_env.pedestrian_list:
                if ped.terminated:
                    removed_ID.append(ped.ID)
            self.sim_env.remove_pedestrian(removed_ID)
            if self.rolling:
                # add in new pedestrian
                for _ in range(len(removed_ID)):
                    self.add_pedestrian()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.sim_env.find_next_position()
            self.sim_env.update()
            self.sim_env.display()
            self.display_destination()
            self.sim_env.flip()

def test1(debug_mode = []):
    print "#################TEST1#################"
    print "# MULTIPLE DESTINATION SIMPLE TEST"
    print "#######################################"
    parameters = {
            'screen_size': (800, 600),
            'screen_name': 'Multiple Destination Test 1',
            'frame_per_second': 2.5,
            'destination_count': 5,
            'max_pedestrian_count': 10,
            'sample_count': 100,
            'rolling': False
            }
    multiple_destination = Multiple_Destination(debug_mode, parameters)
    multiple_destination.run()

def test2(debug_mode):
    print "#################TEST2#################"
    print "# MULTIPLE DESTINATION ROLLING TEST"
    print "#######################################"
    parameters = {
            'screen_size': (800, 600),
            'screen_name': 'Multiple Destination Test 1',
            'frame_per_second': 2.5,
            'destination_count': 5,
            'max_pedestrian_count': 10,
            'sample_count': 100,
            'rolling': True
            }
    multiple_destination = Multiple_Destination(debug_mode, parameters)
    multiple_destination.run()


#test1([1,2,3])
test2([1,2,3])
