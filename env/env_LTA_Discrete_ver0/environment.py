from sim_LTA.simulation import LTA
from agent import Agent
from observation_space import Observation_Space_LTA_Continuous_ver0
from action_space.discrete_action_space import Discrete_Action_Space
from base_environment import Environment
import pygame
import numpy as np
import random
import time
import sys

class LTA_Discrete_ver0(Environment):
    """
    *Dynamic Simulation Model:     LTA
    *Action Space:                 Discrete
    *Observation Style:            Fully Observable
    *Reward Function:              Social Energy
    *Agent Initialization:         Randomly Initialize
    *Pedestrian Initialization:    Randomly Initialize

    action: 0~24, represent accelerations, 0 is no acceleration, 1~24 contains 8 directions and 3 value
            1~3: front low, front middle, front high
            the rest is same style, direction odered clockwise with equal gap (45 degrees)
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

        self._time_penalty_hyperparameter = 0.5
        self._max_ped_count = 40
        self._init_ped_count = 20
        self._add_ped_freq = 4.
        self._rolling = True
        self._pixel2meters = 0.02
        self._expected_speed_sample_func = self._default_expected_speed_sample_func
        self._destination_sample_func = self._default_destination_sample_func
        self._initial_velocity_sample_func = self._default_initial_velocity_sample_func
        self._initial_position_sample_func = self._default_initial_position_sample_func
        self._new_velocity_sample_func = self._default_new_velocity_sample_func
        self._new_position_sample_func = self._default_new_position_sample_func
        self._new_destination_sample_func = self._default_destination_sample_func
        self._new_expected_speed_sample_func = self._default_expected_speed_sample_func
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
                'initial_position_generater': self._initial_position_sample_func,
                'new_position_generater': self._new_position_sample_func,
                'new_velocity_generater': self._new_velocity_sample_func,
                'new_goal_position_generater': self._new_destination_sample_func,
                'new_expected_speed_generater': self._new_expected_speed_sample_func
                }

        self._action_space = Discrete_Action_Space(25)
        self._observation_space =  Observation_Space_LTA_Continuous_ver0()

    def _default_destination_sample_func(self):
        """
        Default function to sample destination position for initial pedestrians
        """
        destination = None
        try:
            destination = random.choice(self._destination_list)
        except AttributeError:
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
        Default function to sample expected speed for initial pedestrians
        """
        return random.uniform(40, 80)

    def _default_initial_velocity_sample_func(self):
        """
        Default function to sample initial velocity for initial pedestrians
        """
        initial_speed = random.gauss(60, 20)
        initial_theta = random.uniform(0, np.pi * 2)
        initial_velocity = initial_speed * np.array([np.cos(initial_theta), np.sin(initial_theta)])
        return initial_velocity

    def _default_initial_position_sample_func(self):
        """
        Default function to sample initial position for initial pedestrians
        """
        ped_position_list = self._sim.get_ped_state()['ped_position']
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

    def _default_new_velocity_sample_func(self):
        """
        Default function to sample initial velocity for new added pedestrians
        """
        return self._default_initial_velocity_sample_func()

    def _default_new_position_sample_func(self):
        """
        Default function to sample initial position for new added pedestrians
        """
        source = random.choice(['up','down', 'left','right'])
        initial_position = np.array([random.uniform(0, self._screen_size[0]), 
            random.uniform(0, self._screen_size[1])])
        if source == 'up':
            initial_position[1] = random.uniform(-100, 0)
        if source == 'down':
            initial_position[1] = random.uniform(0, 100) + self._screen_size[1]
        if source == 'left':
            initial_position[0] = random.uniform(-100, 0)
        if source == 'right':
            initial_position[0] = random.uniform(0, 100) + self._screen_size[0]
        return initial_position

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
            params['lambda2'] = self._ped_params['lambda2']
            params['sigma_d'] = self._ped_params['sigma_d']
            params['sigma_w'] = self._ped_params['sigma_w']
            params['beta'] = self._ped_params['beta']
            params['alpha'] = self._ped_params['alpha']
            params['pixel2meters'] = self._ped_params['pixel2meters']
            params['expected_speed'] = self._ped_params['expected_speed_generater']()
            params['goal_position'] = self._ped_params['goal_position_generater']()
            params['initial_velocity'] = self._ped_params['initial_velocity_generater']()
            params['initial_position'] = self._ped_params['initial_position_generater']()
            
            self._next_ID += 1
            self._sim.add_ped(params)
        
        ped_state = self._sim.get_ped_state()
        agent_state = self._agent.get_agent_state()
        state = ped_state.copy()
        state.update(agent_state)
        observation = state # fully observable

        if self._display_environment:
            self._display(ped_state)
        return observation

    def step(self, action):
        """
        Agent take an action and interact with the environment, will update screen if display is required
        Argument:
            action: 0~24, represent accelerations, 0 is no acceleration, 1~24 contains 8 directions and 3 value
                    1~3: front low, front middle, front high
                    the rest is same style, direction odered clockwise with equal gap (45 degrees)
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
        if action not in self._action_space.range:
            raise ValueError("action is not in range")

        # add new ped if conditioned satisfied
        if random.random() < 1./(self._add_ped_freq + 1e-10) and self._rolling:
            if self._max_ped_count > self._sim.get_ped_count():
                params = {}
                params['ID'] = self._next_ID
                params['lambda1'] = self._ped_params['lambda1']
                params['lambda2'] = self._ped_params['lambda2']
                params['sigma_d'] = self._ped_params['sigma_d']
                params['sigma_w'] = self._ped_params['sigma_w']
                params['beta'] = self._ped_params['beta']
                params['alpha'] = self._ped_params['alpha']
                params['pixel2meters'] = self._ped_params['pixel2meters']
                params['expected_speed'] = self._ped_params['new_expected_speed_generater']()
                params['goal_position'] = self._ped_params['new_goal_position_generater']()
                params['initial_velocity'] = self._ped_params['new_velocity_generater']()
                params['initial_position'] = self._ped_params['new_position_generater']()
                
                self._next_ID += 1
                self._sim.add_ped(params)

        # move the pedestrians first
        self._sim.move()
        self._agent.move(action)
        agent_state = self._agent.get_agent_state()
        ped_state = self._sim.get_ped_state()
        state = ped_state.copy()
        state.update(agent_state)
        reward = self._reward(state)
        done = self._agent.is_done()
        obs = state # fully observable
        
        if self._display_environment:
            #self._clock.tick(self._frame_per_second)
            self._display(ped_state)
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._display_environment = False
                    pygame.display.quit()
                    sys.exit()
                    break
                    
        if done:
            pygame.display.quit()
            pygame.quit()
        return obs, reward, done

    def _reward(self, state):
        """
        The reward for the environment in current situation
        The reward of state of pedestrian itself +
        The penalty of the distance between pedestrians +
        The time penalty
        Input:
            state: dictionary;
                agent_ID: -1
                agent_position: np.1darray; the position of agent
                agent_velocity: np.1darray; the velocity of agent
                agent_goal_position: np.1darray; the goal position of agent
                ped_ID: np.1darray int8; ID of other pedestrians
                ped_position: np.2darray float32; axis 0 is agent index, axis 1 is agent position
                ped_velocity: np.2darray float32; axis 0 is agent index, axis 1 is agent velocity
        Return:
            reward: float;
        """
        l1 = self._ped_params['lambda1']
        l2 = self._ped_params['lambda2']
        sd = self._ped_params['sigma_d']
        sw = self._ped_params['sigma_w']
        beta = self._ped_params['beta']
        alpha = self._ped_params['alpha']
        p2m = self._ped_params['pixel2meters']
        
        ped_velocity = state['ped_velocity']
        ped_position = state['ped_position']
        v = state['agent_velocity'] * p2m
        p = state['agent_position'] * p2m

        E_sum = 0
        for i in range(len(ped_velocity)):
            v2 = ped_velocity[i] * p2m
            p2 = ped_position[i] * p2m
            k = p - p2
            q = v - v2
            t = -np.dot(k, q) / np.linalg.norm(q) ** 2
            d = k + q * max(t, 0)
            dsquare = np.linalg.norm(d) ** 2
            E = np.exp(-dsquare / (2 * sd ** 2))
            wd = np.exp(-np.linalg.norm(k)**2 / (2 * sw**2))
            cos = -np.dot(k, v) / (np.linalg.norm(k) * np.linalg.norm(v))
            wphi = ((1 + cos) / 2)**beta
            E_sum += E * wd * wphi

        reward_params = {
                'pixel2meters': p2m,
                'lambda1': l1
                }
        E_sum += self._agent.reward(reward_params)
        # 0.5 is the time penalty
        return - E_sum - self._time_penalty_hyperparameter

    def display(self):
        """
        Set the function to display mode
        """
        self._screen = pygame.display.set_mode(self._screen_size)
        self._screen_color = (255,255,255)
        pygame.display.set_caption('LTA_Continuous')
        self._screen.fill(self._screen_color)
        #self._clock = pygame.time.Clock()
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
        current_time = time.time()
        try:
            time_gap = current_time - self._last_time
            self._last_time = current_time
            if time_gap < self._step_time:
                time.sleep(self._step_time - time_gap)
        except AttributeError:
            self._last_time = time.time()

        self._screen.fill(self._screen_color)
        ped_position = ped_state['ped_position']
        ped_end_position = ped_state['ped_velocity'] * self._step_time + ped_position
        for ped_index in range(ped_state['ped_ID'].shape[0]):
            position = ped_position[ped_index]
            end_position = ped_end_position[ped_index]
            display_position = (int(position[0]), int(position[1]))
            pygame.draw.circle(self._screen, (0,0,0), display_position, 3, 0)
            end_display_position = (int(end_position[0]), int(end_position[1]))
            pygame.draw.line(self._screen, (0,0,0), display_position, end_display_position, 1)

        agent_position = self._agent.position
        agent_display_position = (int(agent_position[0]), int(agent_position[1]))
        end_position = self._agent.velocity * self._step_time + agent_position
        end_display_position = (int(end_position[0]), int(end_position[1]))
        
        pygame.draw.circle(self._screen, (255,0,0), agent_display_position, 3, 0)
        pygame.draw.line(self._screen, (0,0,0), agent_display_position, end_display_position, 1)

        goal_position = self._agent.goal_position
        goal_display_position = (int(goal_position[0]), int(goal_position[1]))
        pygame.draw.circle(self._screen, (0,0,255), goal_display_position, 6, 0)

    def set_params(self, params = {}):
        """
        set parameters, below parameters are available:
        Input:
            params: dictionary; can partially set
                agent_initial_position
                agent_initial_velocity
                agent_goal_position
                agent_expected_speed
                time_penalty_hyperparameter
                max_ped_count
                init_ped_count
                add_ped_freq
                rolling
                pixel2meters
                expected_speed_sample_func
                destination_sample_func
                initial_velocity_sample_func
                initial_position_sample_func
                new_velocity_sample_func
                new_position_sample_func
                new_destination_sample_func
                new_expected_speed_sample_func
        """
        # set agent parameters
        agent_params = {}
        if 'agent_initial_position' in params:
            agent_params['initial_position'] = params['agent_initial_position']
        if 'agent_initial_velocity' in params:
            agent_params['initial_velocity'] = params['agent_initial_velocity']
        if 'agent_goal_position' in params:
            agent_params['default_goal_position'] = params['agent_goal_position']
        if 'agent_expected_speed' in params:
            agent_params['default_expected_speed'] = params['agent_expected_speed']
        self._agent.set_params(agent_params)
            
        # set other parameters in the environment
        if 'time_penalty_hyperparameter' in params:
            self._time_penalty_hyperparameter = params['time_penalty_hyperparameter']
        if 'max_ped_count' in params:
            self._max_ped_count = params['max_ped_count']
        if 'init_ped_count' in params:
            self._init_ped_count = params['init_ped_count']
        if 'add_ped_freq' in params:
            self._add_ped_freq = params['add_ped_freq']
        if 'rolling' in params:
            self._rolling = params['rolling']
        if 'pixel2meters' in params:
            self._pixel2meters = params['pixel2meters']
        if 'expected_speed_sample_func' in params:
            self._ped_params['expected_speed_generater'] = params['expected_speed_sample_func']
        if 'destination_sample_func' in params:
            self._ped_params['goal_position_generater'] = params['destination_sample_func']
        if 'initial_velocity_sample_func' in params:
            self._ped_params['initial_velocity_generater'] = params['initial_velocity_sample_func']
        if 'initial_position_sample_func' in params:
            self._ped_params['initial_position_generater'] = params['initial_position_sample_func']
        if 'new_velocity_sample_func' in params:
            self._ped_params['new_velocity_generater'] = params['new_velocity_sample_func']
        if 'new_position_sample_func' in params:
            self._ped_params['new_position_generater'] = params['new_position_sample_func']
        if 'new_destination_sample_func' in params:
            self._ped_params['new_goal_position_generater'] = params['new_destination_sample_func']
        if 'new_expected_speed_sample_func' in params:
            self._ped_params['new_expected_speed_generater'] = params['new_expected_speed_sample_func']

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



