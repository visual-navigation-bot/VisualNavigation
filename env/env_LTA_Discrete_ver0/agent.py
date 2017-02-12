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
        self._default_expected_speed = None
        self._speed_magnitude = [15., 25., 35.]

        self._position = None
        self._velocity = None
        self._expected_speed = None
        self._goal_position = None
        self._ID = -1

    def move(self, action):
        """
        Move the agent as the action by step time
        Input:
            action: 0~24, represent acceleration, 0 is no acceleration, 1~24 contains 8 directions and 3 value
                    1~3: front low, front middle, front high
                    the rest is same style, direction odered clockwise with equal gap (45 degrees)
        """
        if action == 0:
            return
        direction = (action - 1) / 3
        magnitude = self._speed_magnitude[action - 1 - 3 * direction]
        theta = direction * np.pi / 4.
        acceleration = np.array([magnitude * np.cos(theta), magnitude * np.sin(theta)])
        old_velocity = self._velocity.copy()
        self._velocity += acceleration * self._step_time
        self._position += (self._velocity + old_velocity) * self._step_time / 2

    def get_agent_state(self):
        """
        Return the state of the agent
        Return:
            agent_state: dictionary;
                agent_ID: -1
                agent_position: np1darray; position of agent
                agent_velocity: np1darray; velocity of agent
                agent_goal_position: np1darray; goal position of agent
        """
        agent_state = {
                'agent_ID': -1,
                'agent_position': self._position,
                'agent_velocity': self._velocity,
                'agent_goal_position': self._goal_position
                }
        return agent_state

    def reset(self):
        """
        reset the position and velocity of pedestrian
        No Input and Return value
        """
        if self._initial_velocity is None:
            self._random_initialize_velocity()
        else:
            self._velocity = self._initial_velocity

        if self._initial_position is None:
            self._random_initialize_position()
        else:
            self._position = self._initial_position

        if self._default_goal_position is None:
            self._random_set_goal_position()
        else:
            self._goal_position = self._default_goal_position

        if self._default_expected_speed is None:
            self._random_set_expected_speed()
        else:
            self._expected_speed = self._default_expected_speed

    def set_params(self, params = {}):
        """
        set initial conditions
        Input:
            params: dictionary;
                initial_position: np1darray; initial position of agent
                initial_velocity: np1darray; initial velocity of agent
                default_goal_position: np1darray; goal position of agent
                default_expected_speed: float; expected speed of agent
        """
        if 'initial_velocity' in params:
            self._initial_velocity = params['initial_velocity']
        if 'initial_position' in params:
            self._initial_position = params['initial_position']
        if 'default_goal_position' in params:
            self._default_goal_position = params['default_goal_position']
        if 'default_expected_speed' in params:
            self._default_expected_speed = params['default_expected_speed']

    def is_done(self):
        """
        Return:
            done: bool; if the agent arrived its destination
        """
        return np.linalg.norm(self._position - self._goal_position) < 10

    def reward(self, reward_params):
        """
        Return the reward of the agent, get by its expected speed not reached
        Input: 
            reward_params: dictionary;
                pixel2meters: float; one pixel is how many meters
                lambda1: float;
        Return:
            reward: float;
        """
        p2m = reward_params['pixel2meters']
        l1 = reward_params['lambda1']
        speed_diff = self._expected_speed - np.linalg.norm(self._velocity)
        return l1 * (p2m * speed_diff)**2



    def _random_initialize_position(self):
        self._position = np.array(
                [random.uniform(0, self._field_size[0]), random.uniform(0, self._field_size[1])]
                )

    def _random_initialize_velocity(self):
        expected_speed = random.uniform(40., 80.)
        direction = random.uniform(0, 2 * np.pi)
        self._velocity = np.array([np.cos(direction), np.sin(direction)]) * expected_speed

    def _random_set_expected_speed(self):
        self._expected_speed = random.uniform(40., 80.)

    def _random_set_goal_position(self):
        self._goal_position = np.array(
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

    @property
    def expected_speed(self):
        return self._expected_speed
