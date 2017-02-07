from simulation import LTA
import pygame
import numpy as np
import random

class LTA_Continuous_v0:
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
        self._screen_size = field_size
        self._screen = pygame.display.set_mode(self._screen_szie)
        self._screen_color = (255,255,255)
        pygame.display.set_caption('LTA_Continuous')
        self._screen.fill(self._screen_color)
        self._clock = pygame.time.Clock()
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

        self._action_space = None # an object
        self._observation_space = None # an object

    def _default_destination_sample_func(self):
        try:
            a = 2
        except NameError:

        return
    def _default_expected_speed_sample_func(self):
        return
    def _default_initial_velocity_sample_func(self):
        return
    def _default_initial_position_sample_func(self):
        return

    def step(self, action):
        """
        Agent take an action and interact with the environment
        Argument:
            - action
        Return:
            - observation
            - reward
            - done
        """

    def display(self):
        """
        Display the environment
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the environment
        Return:
            observation: dictionary;
                agent_ID: -1
                agent_position: np.1darray; the position of agent
                agent_velocity: np.1darray; the velocity of agent
                ped_ID: np.1darray int8; ID of other pedestrians
                ped_position: np.2darray float32; axis 0 is agent index, axis 1 is agent position
                ped_velocity: np.2darray float32; axis 0 is agent index, axis 1 is agent velocity
        """
        self._agent.reset()
        

        raise NotImplementedError

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
        print "further description please refer to the source code"

class Agent:
    """
    Class Agent to add into LTA model
    """
    def __init__(self, step_time, field_size):
        """
        The Agent of this environment
        Input:
            field_size: (int, int); the field size
            step_time: float; the step time
        """
        self._step_time = step_time
        self._field_size = field_size

        self._initial_velocity = None
        self._initial_position = None
        self._default_goal_position = None

        self._position = None
        self._velocity = None
        self._goal_position = None
        self._ID = -1
        
    def get_agent_state(self):
        """
        Return the state of agent
        Input:
            None
        Return:
            params: dictionary;
                position: np1darray; the position of agent
                velocity: np1darray; the velocity of agent
                goal_position: np1darray; the goal position of agent
                ID: int; the ID of agent, should be -1
        """
        params = {}
        params['position'] = self._position
        params['velocity'] = self._velocity
        params['goal_position'] = self._goal_position
        params['ID'] = self._ID
        return params

    def move(self, action):
        """
        Move the agent as the action by step time
        """


    def reset(self):
        """
        reset the position and velocity of pedestrian
        No Input and Return value
        """
        if self._initial_velocity is None:
            self._random_initialize_position()
        else:
            self._position = self._initial_position

        if self._initial_position is None:
            self._random_initialize_position()
        else:
            self._velocity = self._initial_velocity

        if self._default_goal_position is None:
            self._random_set_goal_position()
        else:
            self._goal_position = self._default_goal_position

    def set_params(self, params):
        """
        set initial conditions
        Input:
            params: dictionary;
                initial_position: np1darray; initial position of agent
                initial_velocity: np1darray; initial velocity of agent
                default_goal_position: np1darray; goal position of agent
        """
        self._initial_velocity = params['initial_velocity']
        self._initial_position = params['initial_position']
        self._default_goal_position = params['default_goal_position']


    def _random_initialize_position(self):
        self._position = np.array(
                [random.uniform(0, self._field_size[0]), random.uniform(0, self._field_size[1])]
                )

    def _random_initialize_velocity(self):
        expected_speed = random.uniform(40., 80.)
        self._velocity = np.array([random.random(), random.random()]) * expected_speed

    def _random_set_goal_position(self):
        self._position = np.array(
                [random.uniform(0, self._field_size[0]), random.uniform(0, self._field_size[1])]
                )

    @property
    def position(self):
        return self._position

    @property
    def velocity(self):
        return self._velocity

    @property
    def goal_position(self):
        return self._goal_position

    @property
    def ID(self):
        return self._ID


