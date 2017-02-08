import numpy as np
import random

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
