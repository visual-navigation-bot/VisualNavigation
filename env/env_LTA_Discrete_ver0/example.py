import environment
import numpy as np
import random
import time

def Opposite_Walk(step_time, field_size, ow_params = {}):
    print "==========LTA CONTINUOUS VER0 OPPOSITE WALK ENVIRONMENT==========="
    fps = 1./step_time

    class Agent_State_Initializer:
        def __init__(self, seed):
            """
            Initialize agent state
            Input:
                seed: int; seed of following
            """
            self._seed = seed
            random.seed(seed)
            position_x = 50 + 150 * random.random()
            position_y = 300 + 50 * random.uniform(-1,1)
            self._initial_position = np.array([position_x, position_y])

            self._expected_speed = random.uniform(-20,20) + 60.
            
            init_speed_diff = 10 * random.uniform(-1,1)
            init_speed = self._expected_speed + init_speed_diff
            init_direction = 0.2 * random.uniform(-1,1)
            self._initial_velocity = np.array([init_speed * np.cos(init_direction), 
                init_speed * np.sin(init_direction)])

            position_x = 750 - 150 * random.random()
            position_y = 300 + 50 * random.uniform(-1,1)
            random.seed(time.time())
            self._goal_position = np.array([position_x, position_y])

        @property
        def initial_position(self):
            return self._initial_position

        @property
        def initial_velocity(self):
            return self._initial_velocity

        @property
        def expected_speed(self):
            return self._expected_speed

        @property
        def goal_position(self):
            return self._goal_position

    class Ped_State_Initializer:
        def __init__(self, seed):
            self._seed = seed
            # expected speed, goal pos, init vel, init pos
            self._function_called = np.array([True, True, True, True])
            self._expected_speed = None
            self._goal_position = None
            self._initial_velocity = None
            self._initial_position = None
            self._initialize_ped_state()

        def _initialize_ped_state(self):
            if np.prod(self._function_called):
                # initialize stuff
                random.seed(self._seed)
                position_x = 750 - 50 * random.random()
                position_y = 300 + 250 * random.uniform(-1,1)
                self._initial_position = np.array([position_x, position_y])

                self._expected_speed = random.uniform(-20,20) + 60.
                
                init_speed_diff = 10 * random.uniform(-1,1)
                init_speed = self._expected_speed + init_speed_diff
                init_direction = np.pi + 0.2 * random.uniform(-1,1)
                self._initial_velocity = np.array([init_speed * np.cos(init_direction), 
                    init_speed * np.sin(init_direction)])

                position_x = 50 + 150 * random.random()
                position_y = 300 + 250 * random.uniform(-1,1)
                self._goal_position = np.array([position_x, position_y])

                self._function_called = np.array([False, False, False, False])
                self._seed = self._seed * 1.1 + 0.1
                random.seed(time.time())

        def expected_speed_sample_func(self):
            self._initialize_ped_state()
            self._function_called[0] = True
            return self._expected_speed

        def destination_sample_func(self):
            self._initialize_ped_state()
            self._function_called[1] = True
            return self._goal_position

        def initial_velocity_sample_func(self):
            self._initialize_ped_state()
            self._function_called[2] = True
            return self._initial_velocity

        def initial_position_sample_func(self):
            self._initialize_ped_state()
            self._function_called[3] = True
            return self._initial_position

    class New_Ped_State_Initializer:
        def __init__(self, seed):
            self._seed = seed
            # expected speed, goal pos, init vel, init pos
            self._function_called = np.array([True, True, True, True])
            self._expected_speed = None
            self._goal_position = None
            self._initial_velocity = None
            self._initial_position = None
            self._initialize_ped_state()

        def _initialize_ped_state(self):
            if np.prod(self._function_called):
                # initialize stuff
                random.seed(self._seed)
                position_x = 800 + 50 * random.random()
                position_y = 300 + 250 * random.uniform(-1,1)
                self._initial_position = np.array([position_x, position_y])

                self._expected_speed = random.uniform(-20,20) + 60.
                
                init_speed_diff = 10 * random.uniform(-1,1)
                init_speed = self._expected_speed + init_speed_diff
                init_direction = np.pi + 0.2 * random.uniform(-1,1)
                self._initial_velocity = np.array([init_speed * np.cos(init_direction), 
                    init_speed * np.sin(init_direction)])

                position_x = 50 + 150 * random.random()
                position_y = 300 + 250 * random.uniform(-1,1)
                self._goal_position = np.array([position_x, position_y])

                self._function_called = np.array([False, False, False, False])
                self._seed = self._seed * 1.2 + 0.3
                random.seed(time.time())

        def new_expected_speed_sample_func(self):
            self._initialize_ped_state()
            self._function_called[0] = True
            return self._expected_speed

        def new_destination_sample_func(self):
            self._initialize_ped_state()
            self._function_called[1] = True
            return self._goal_position

        def new_initial_velocity_sample_func(self):
            self._initialize_ped_state()
            self._function_called[2] = True
            return self._initial_velocity

        def new_initial_position_sample_func(self):
            self._initialize_ped_state()
            self._function_called[3] = True
            return self._initial_position


    agent_state_seed = 10478
    asi = Agent_State_Initializer(agent_state_seed)

    agent_initial_position = asi.initial_position
    agent_initial_velocity = asi.initial_velocity
    agent_goal_position = asi.goal_position
    agent_expected_speed = asi.expected_speed

    ped_state_seed = 25.4
    psi = Ped_State_Initializer(ped_state_seed)

    ped_state_seed = 789.3
    npsi = New_Ped_State_Initializer(ped_state_seed)
    
    time_penalty_hyperparameter = 0.0
    max_ped_count = 20
    init_ped_count = 15
    add_ped_freq = 1
    rolling = True
    pixel2meters = 0.02

    env_params = {
            'agent_initial_position': agent_initial_position,
            'agent_initial_velocity': agent_initial_velocity,
            'agent_goal_position': agent_goal_position,
            'agent_expected_speed': agent_expected_speed,
            'time_penalty_hyperparameter': time_penalty_hyperparameter,
            'max_ped_count': max_ped_count,
            'init_ped_count': init_ped_count,
            'add_ped_freq': add_ped_freq,
            'rolling': rolling,
            'pixel2meters': pixel2meters,
            'expected_speed_sample_func': psi.expected_speed_sample_func,
            'destination_sample_func': psi.destination_sample_func,
            'initial_velocity_sample_func': psi.initial_velocity_sample_func,
            'initial_position_sample_func': psi.initial_position_sample_func,
            'new_expected_speed_sample_func': npsi.new_expected_speed_sample_func,
            'new_destination_sample_func': npsi.new_destination_sample_func,
            'new_velocity_sample_func': npsi.new_initial_velocity_sample_func,
            'new_position_sample_func': npsi.new_initial_position_sample_func
            }

    env = environment.LTA_Discrete_ver0(0.4, (800,600))
    env.set_params(env_params)
    env.set_params(ow_params)
    return env






