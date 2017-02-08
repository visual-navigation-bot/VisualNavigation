from sim_LTA.LTA import LTA
from agent import Agent
from observation_space import Observation_Space_LTA_Continuous_ver0
from action_space.continuous_action_space import Continuous_Action_Space
from base_environment import Environment
import pygame
import numpy as np
import random

class LTA_Continuous_ver0(Environment):
    """
    *Dynamic Simulation Model:     LTA
    *Action Space:                 Continuous
    *Observation Style:            Fully Observable
    *Reward Function:              Social Energy
    *Agent Initialization:         Randomly Initialize
    *Pedestrian Initialization:    Randomly Initialize

    action: np.1darray; the acceleration of the agent (x,y)
    observation: dictionary;
        agent_ID: -1
        agent_position: np.1darray; the position of agent
        agent_velocity: np.1darray; the velocity of agent
        agent_goal_position: np.1darray; the goal position of agent
        ped_ID: np.1darray int8; ID of other pedestrians
        ped_position: np.2darray float32; axis 0 is agent index, axis 1 is agent position
        ped_velocity: np.2darray float32; axis 0 is agent index, axis 1 is agent velocity
    """
    def __init__(self, step_time, field_size):
        """
        Input:
            step_time: float; time interval of taking steps
            field_size: (int, int); the size of the field
        """
        Environment.__init__(self, step_time, field_size)
        self._screen_size = field_size
        self._display_environment = False
        self._step_time = step_time
        self._frame_per_second = 1. / step_time

        self._agent = Agent(step_time, field_size)

        self._sim = LTA(field_size, self._frame_per_second)
        self._next_ID = 0

        self._max_ped_count = 20
        self._init_ped_count = 10
        self._rolling = True
        self._pixel2meters = 0.02
        self._expected_speed_sample_func = self._default_expected_speed_sample_func
        self._destination_sample_func = self._default_destination_sample_func
        self._initial_velocity_sample_func = self._default_initial_velocity_sample_func
        self._initial_position_sample_func = self._default_initial_position_sample_func
        self._ped_params = {
                'lambda1' : 2.33,
                'lambda2' : 2.073,
                'sigma_d' : 0.361,
                'sigma_w' : 2.088,
                'beta' : 1.462,
                'alpha' : 0.730,
                'pixel2meters' : self._pixel2meters,
                'expected_speed_generater': self._expected_speed_sample_func,
                'goal_position_generater': self._destination_sample_func,
                'initial_velocity_generater': self._initial_velocity_sample_func,
                'initial_position_generater': self._initial_position_sample_func
                }

        self._action_space = Continuous_Action_Space(np.array([-1000.,-1000.]), np.array([1000.,1000.]))
        self._observation_space =  Observation_Space_LTA_Continuous_ver0()

    def _default_destination_sample_func(self):
        """
        Default function to sample destination position
        """
        destination = None
        try:
            destination = random.choice(self._destination_list)
        except NameError:
            # destination list is not created yet
            destination_count = 4
            self._destination_list = []
            while len(self._destination_list) < destination_count:
                new_destination = np.array([random.uniform(0, self._screen_size[0]), 
                    random.uniform(0, self._screen_size[1])])
                allowed = True
                for destination in self._destination_list:
                    # minimum destination distance is 20 pixels
                    if np.linalg.norm(new_destination - destination) < 20:
                        allowed = False
                if allowed:
                    self._destination_list.append(new_destination)
            destination = random.choice(self._destination_list)
        return destination

    def _default_expected_speed_sample_func(self):
        """
        Default function to sample expected speed
        """
        return random.uniform(40, 80)

    def _default_initial_velocity_sample_func(self):
        """
        Default function to sample initial velocity
        """
        initial_speed = random.gauss(60, 20)
        initial_theta = random.uniform(0, np.pi * 2)
        initial_velocity = initial_speed * np.array([np.cos(initial_theta), np.sin(initial_theta)])
        return initial_velocity

    def _default_initial_position_sample_func(self):
        """
        Default function to sample initial position
        """
        ped_position_list = self._sim.get_ped_state['ped_position']
        initial_position = None
        allowed = False
        while not allowed:
            allowed = True
            initial_position = np.array([random.uniform(0, self._screen_size[0]), 
                    random.uniform(0, self._screen_size[1])])
            if np.linalg.norm(initial_position - self._agent.position) < 10:
                allowed = False
            for ped_index in range(ped_position_list.shape[0]):
                if np.linalg.norm(initial_position - ped_position_list[ped_index]) < 10:
                    allowed = False
        return initial_position

    def step(self, action):
        """
        Agent take an action and interact with the environment, will update screen if display is required
        Argument:
            action: np1darray; the acceleration of the agent
        Return:
            observation: dictionary;
                agent_ID: -1
                agent_position: np.1darray; the position of agent
                agent_velocity: np.1darray; the velocity of agent
                agent_goal_position: np.1darray; the goal position of agent
                ped_ID: np.1darray int8; ID of other pedestrians
                ped_position: np.2darray float32; axis 0 is agent index, axis 1 is agent position
                ped_velocity: np.2darray float32; axis 0 is agent index, axis 1 is agent velocity
            reward: float; the reward function
            done: bool; if the process is done or not
        """
        ped_state = self._sim.get_ped_state()
        obs, reward, done = self._step(ped_state, action)
        if self._display_environment:
            self._clock.tick(self._frame_per_second)
            self._screen.fill(self._screen_color)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._display_environment = False
                    pygame.display.quit()
                    pygame.quit()
            self._display(ped_state)
            pygame.display.flip()
        if done:
            pygame.display.quit()
            pygame.quit()
        return obs, reward, done

    def _step(self, ped_state, action):
        """
        Agent take an action and interact with the environment
        Argument:
            ped_state: dictionary;
                ped_ID: np.1darray int8; ID of other pedestrians
                ped_position: np.2darray float32; axis 0 is agent index, axis 1 is agent position
                ped_velocity: np.2darray float32; axis 0 is agent index, axis 1 is agent velocity
            action: np1darray; the acceleration of the agent
        Return:
            observation: dictionary;
                agent_ID: -1
                agent_position: np.1darray; the position of agent
                agent_velocity: np.1darray; the velocity of agent
                agent_goal_position: np.1darray; the goal position of agent
                ped_ID: np.1darray int8; ID of other pedestrians
                ped_position: np.2darray float32; axis 0 is agent index, axis 1 is agent position
                ped_velocity: np.2darray float32; axis 0 is agent index, axis 1 is agent velocity
            reward: float; the reward function
            done: bool; if the process is done or not
        """

            

    def display(self):
        """
        Set the function to display mode
        """
        self._screen = pygame.display.set_mode(self._screen_szie)
        self._screen_color = (255,255,255)
        pygame.display.set_caption('LTA_Continuous')
        self._screen.fill(self._screen_color)
        self._clock = pygame.time.Clock()
        self._display_environment = True

    def _display(self, ped_state):
        """
        put all objects on the display screen
        Input:
            ped_state: dictionary;
                ped_ID: np.1darray int8; ID of other pedestrians
                ped_position: np.2darray float32; axis 0 is agent index, axis 1 is agent position
                ped_velocity: np.2darray float32; axis 0 is agent index, axis 1 is agent velocity
        """
        ped_position = ped_state['ped_position']
        ped_end_position = ped_state['ped_velocity'] * self._time_step + ped_position
        for ped_index in range(ped_state['ped_ID'].shape[0]):
            position = ped_position[ped_index]
            end_position = ped_end_position[ped_index]
            display_position = (int(position[0]), int(position[1]))
            pygame.draw.circle(self._screen, (0,0,0), display_position, 3, 0)
            end_display_position = (int(end_position[0]), int(end_position[1]))
            pygame.draw.line(self.screen, (0,0,0), display_position, end_display_position, 1)

        agent_position = self._agent.position
        agent_display_position = (int(agent_position[0]), int(agent_position[1]))
        end_position = self._agent.velocity * self._time_step + agent_position
        end_display_position = (int(end_position[0]), int(end_position[1]))
        
        pygame.draw.circle(self._screen, (255,0,0), agent_display_position, 3, 0)
        pygame.draw.line(self.screen, (0,0,0), agent_display_position, end_display_position, 1)

        goal_position = self._agent.goal_position
        goal_display_position = (int(goal_position[0]), int(goal_position[1]))
        pygame.draw.circle(self._screen, (0,0,255), goal_display_position, 6, 0)

    def reset(self):
        """
        Reset the environment
        Return:
            observation: dictionary;
                agent_ID: -1
                agent_position: np.1darray; the position of agent
                agent_velocity: np.1darray; the velocity of agent
                agent_goal_position: np.1darray; the goal position of agent
                ped_ID: np.1darray int8; ID of other pedestrians
                ped_position: np.2darray float32; axis 0 is agent index, axis 1 is agent position
                ped_velocity: np.2darray float32; axis 0 is agent index, axis 1 is agent velocity
        """
        self._agent.reset()
        self._sim.add_agent(self._agent)

        for ped_index in range(self._init_ped_count):
            params = {}
            params['ID'] = self._next_ID
            params['lambda1'] = self._ped_params['lambda1']
            parmas['lambda2'] = self._ped_params['lambda2']
            parmas['sigma_d'] = self._ped_params['sigma_d']
            parmas['sigma_w'] = self._ped_params['sigma_w']
            parmas['beta'] = self._ped_params['beta']
            parmas['alpha'] = self._ped_params['alpha']
            parmas['pixel2meters'] = self._ped_params['pixel2meters']
            params['expected_speed'] = self._ped_params['expected_speed_generater']()
            params['goal_position'] = self._ped_params['goal_position_generater']()
            params['initial_velocity'] = self._ped_params['initial_velocity_generater']()
            params['initial_position'] = self._ped_params['initial_position_generater']()
            
            self._next_ID += 1
            self._sim.add_ped(params)
        
        ped_state = self._sim.get_ped_state()
        observation = {
                'agent_ID': self._agent.ID,
                'agent_position': self._agent.position,
                'agent_velocity': self._agent.velocity,
                'agent_goal_position': self._agent.goal_position,
                'ped_ID': ped_state['ped_ID'],
                'ped_position': ped_state['ped_position'],
                'ped_velocity': ped_state['ped_velocity']
                }
        if self._display_environment:
            self._display(ped_state)
        return observation

    def set_params(self, params = {}):
        """
        set parameters, below parameters are available:
        Input:
            params: dictionary; can partially set
                
        """

    @property
    def action_space(self):
        if self._action_space is not None:
            return self._action_space
        else:
            raise ValueError('Action space is undefined.')
    @property
    def observation_space(self):
        if self._observation_space is not None:
            return self._observation_space
        else:
            raise ValueError('Observation space is undefined.')

    def __str__(self):
        return "further description please refer to the source code"



