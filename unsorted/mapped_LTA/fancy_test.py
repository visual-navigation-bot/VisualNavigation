import pygame
import simulation
import numpy as np
import random

class Multiple_Destination:
    def __init__(self, debug_mode, parameters, map_restore_path):
        self.screen_size = parameters['screen_size']
        self.screen_name = parameters['screen_name']
        self.patch_size = parameters['patch_size']
        self.frame_per_second = parameters['frame_per_second']
        self.destination_count = parameters['destination_count']
        self.max_pedestrian_count = parameters['max_pedestrian_count']
        self.pixel2meters = parameters['pixel2meters']
        self.rolling = parameters['rolling']

        self.min_destination_distance = 50.
        self.arrival_distance = 20.
        self.destination_list = []

        self.sim_env = simulation.Simulation_Environment(self.screen_size, 
                self.screen_name, self.frame_per_second)
        self.sim_env.set_debug_mode(debug_mode)
        self.map_restore_path = map_restore_path
        self.sim_env.add_navigation_map(self.patch_size, self.map_restore_path)
        self.energy_map = self.sim_env.energy_map

        self.screen = self.sim_env.screen
        self.next_ID = 0

        # generate pedestrian destinations
        allowed_x, allowed_y = np.where(self.energy_map < 0.2)
        while len(self.destination_list) < self.destination_count:
            index = random.randint(0, len(allowed_x) - 1)
            new_destination = np.array([allowed_x[index], allowed_y[index]])
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
        initial_velocity = np.array([random.random(), random.random()]) * expected_speed
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

        print "ID: ", self.next_ID
        print "expected speed: ", expected_speed
        print "goal position: ", destination
        print "initial velocity: ", initial_velocity
        print "initial position: ", initial_position
        print ""

        parameters = {
                'ID' : self.next_ID,
                'lambda1' : 2.33,
                'lambda2' : 2.073,
                'sigma_d' : 0.361,
                'sigma_w' : 2.088,
                'beta' : 1.462,
                'alpha' : 0.730,
                'pixel2meters' : self.pixel2meters,
                'expected_speed': expected_speed,
                'goal_position': destination,
                'initial_velocity': initial_velocity,
                'initial_position': initial_position
                    }
        ped = simulation.Pedestrian(parameters, self.sim_env)
        self.sim_env.add_pedestrian(ped)
        self.next_ID += 1

    def destination_arrival(self, ped):
        # did the pedestrian arrive its destination
        return np.linalg.norm(ped.goal_position - ped.position) < self.arrival_distance

    def display_destination(self):
        # display destinations
        for destination in self.destination_list:
            position = (int(destination[0]), int(destination[1]))
            pygame.draw.circle(self.screen, (255,0,0), position, 10, 0)

    def run(self):
        running = True
        while running:
            self.sim_env.reset_screen()
            self.sim_env.clock_tick()
            for ped in self.sim_env.pedestrian_list:
                ped_ID = ped.ID
                if self.destination_arrival(ped):
                    self.sim_env.remove_pedestrian([ped_ID])
                    if self.rolling:
                        # add in new pedestrian
                        self.add_pedestrian()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.sim_env.calculate_cross_pedestrian_value()
            self.sim_env.move()
            self.sim_env.display()
            self.display_destination()
            pygame.display.flip()

def test1(debug_mode):
    print "#################TEST1#################"
    print "# MULTIPLE DESTINATION SIMPLE TEST"
    print "#######################################"
    parameters = {
            'screen_size': (600, 400),
            'screen_name': 'Multiple Destination Test 1',
            'patch_size': 10,
            'frame_per_second': 2.5,
            'destination_count': 5,
            'max_pedestrian_count': 40,
            'pixel2meters': 0.02,
            'rolling': False
            }
    map_restore_path = './test1.pkl' 
    multiple_destination = Multiple_Destination(debug_mode, parameters, map_restore_path)
    multiple_destination.run()

def test2(debug_mode):
    print "#################TEST2#################"
    print "# MULTIPLE DESTINATION ROLLING TEST"
    print "#######################################"
    parameters = {
            'screen_size': (600, 400),
            'screen_name': 'Multiple Destination Test 2',
            'patch_size': 1,#10,
            'frame_per_second': 2.5,
            'destination_count': 2,#10,
            'max_pedestrian_count': 10,#40,
            'pixel2meters': 0.02,
            'rolling': True
            }
    map_restore_path = './test1.pkl' 
    multiple_destination = Multiple_Destination(debug_mode, parameters, map_restore_path)
    multiple_destination.run()



#test1() #test1(3)
test2([]) #test2(3)
