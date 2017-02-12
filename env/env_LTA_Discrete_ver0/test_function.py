import environment
import agent
from sim_LTA.simulation import LTA
import observation_space
import numpy as np
import random
import time

def show_observation(observation):
    print "#####Observation#####"
    print "AGENT INFO:"
    print "    position: ",observation['agent_position']
    print "    velocity: ",observation['agent_velocity']
    print "    goal position: ", observation['agent_goal_position']
    print "PEDESTRIAN INFO:"
    for ped_index in range(len(observation['ped_ID'])):
        print "    ID: ", observation['ped_ID'][ped_index]
        print "    position: ", observation['ped_position'][ped_index]
        print "    velocity: ", observation['ped_velocity'][ped_index]
        print ""
    print "#####################"


def show_agent_state(agent_state):
    print "#####AGENT INFO#####"
    print "    position: ",agent_state['agent_position']
    print "    velocity: ",agent_state['agent_velocity']
    print "    goal position: ", agent_state['agent_goal_position']
    print "#####################"

def show_ped_state(ped_state):
    print "#####PED INFO#####"
    for ped_index in range(len(ped_state['ped_ID'])):
        print "    ID: ", ped_state['ped_ID'][ped_index]
        print "    position: ", ped_state['ped_position'][ped_index]
        print "    velocity: ", ped_state['ped_velocity'][ped_index]
        print ""
    print "#####################"



def test1():
    print "==========TEST 1: AGENT FUNCTIONS==========="
    # test agent all functions
    step_time = 0.4
    field_size = (800, 600)
    _agent = agent.Agent(step_time, field_size)
    _agent.reset()
    _agent_state = _agent.get_agent_state()
    show_agent_state(_agent_state)

    print "action is 1"
    action = 2
    _agent.move(action)
    _agent_state = _agent.get_agent_state()
    show_agent_state(_agent_state)

    params = {
            'initial_position': np.array([100., 100.]),
            'initial_velocity': np.array([40.,0.]),
            'default_goal_position': np.array([300., 100.]),
            'default_expected_speed': 30.
            }
    print "set parameters: ",params
    _agent.set_params(params)
    _agent.reset()
    _agent_state = _agent.get_agent_state()
    show_agent_state(_agent_state)

    reward_params = {
            'pixel2meters': 0.5,
            'lambda1': 3
            }
    print "reward: ", _agent.reward(reward_params)
    print "done: ", _agent.is_done()

    params = {
            'initial_position': np.array([100., 100.]),
            'initial_velocity': np.array([40.,0.]),
            'default_goal_position': np.array([105., 100.]),
            'default_expected_speed': 30.
            }
    print "set parameters: ",params
    _agent.set_params(params)
    _agent.reset()
    _agent_state = _agent.get_agent_state()
    show_agent_state(_agent_state)
    print "done: ", _agent.is_done()

    params = {
            'initial_position': np.array([100., 100.]),
            'initial_velocity': np.array([40.,0.]),
            'default_goal_position': np.array([300., 100.]),
            'default_expected_speed': 30.
            }
    print "set parameters: ",params
    _agent.set_params(params)
    _agent.reset()
    _agent_state = _agent.get_agent_state()
    show_agent_state(_agent_state)

    print "action is 24"
    action = 24
    _agent.move(action)
    _agent_state = _agent.get_agent_state()
    show_agent_state(_agent_state)



def test2():
    print "==========TEST 2: TEST SIM_LTA FUNCTION WITH AGENT==========="
    step_time = 0.4
    fps = 1. / step_time
    field_size = (800, 600)
    params = {
            'initial_position': np.array([101., 101.]),
            'initial_velocity': np.array([40.,0.]),
            'default_goal_position': np.array([301., 101.]),
            'default_expected_speed': 30.
            }
    print "agent parameters: ",params
    _agent = agent.Agent(step_time, field_size)
    _agent.set_params(params)
    _agent.reset()
    _agent_state = _agent.get_agent_state()
    show_agent_state(_agent_state)

    print "======= DID NOT ADD PED ======="
    _sim = LTA(field_size, fps)
    _sim.add_agent(_agent)
    ped_state = _sim.get_ped_state()
    show_ped_state(ped_state)
    print "ped count without agent: ", _sim.get_ped_count()
    
    print "======= ADD ONE PED ========"
    parameters = {
            'ID' : 7,
            'lambda1' : 2.33,
            'lambda2' : 2.073,
            'sigma_d' : 0.361,
            'sigma_w' : 2.088,
            'beta' : 1.462,
            'alpha' : 0.730,
            'pixel2meters' : 0.2,
            'expected_speed': 45,
            'goal_position': np.array([500., 100.]),
            'initial_velocity': np.array([45., 0.]),
            'initial_position': np.array([100., 100.])
                }
    _sim.add_ped(parameters)
    ped_state = _sim.get_ped_state()
    show_ped_state(ped_state)
    print "ped count without agent: ", _sim.get_ped_count()
    print "AFTER MOVED"
    _sim.move()
    ped_state = _sim.get_ped_state()
    show_ped_state(ped_state)
    print "ped count without agent: ", _sim.get_ped_count()

    print "======= ONE PED NO AGENT ======="
    _sim = LTA(field_size, fps)
    parameters = {
            'ID' : 7,
            'lambda1' : 2.33,
            'lambda2' : 2.073,
            'sigma_d' : 0.361,
            'sigma_w' : 2.088,
            'beta' : 1.462,
            'alpha' : 0.730,
            'pixel2meters' : 0.2,
            'expected_speed': 45,
            'goal_position': np.array([500., 100.]),
            'initial_velocity': np.array([45., 0.]),
            'initial_position': np.array([100., 100.])
                }
    _sim.add_ped(parameters)
    ped_state = _sim.get_ped_state()
    show_ped_state(ped_state)
    print "ped count without agent: ", _sim.get_ped_count()
    print "AFTER MOVED"
    _sim.move()
    ped_state = _sim.get_ped_state()
    show_ped_state(ped_state)
    print "ped count without agent: ", _sim.get_ped_count()
    print "existance of agent did influence pedestrian movement"


def test3():
    print "==========TEST 3: TEST SIM_LTA FUNCTION WITH AGENT==========="
    step_time = 0.4
    fps = 1. / step_time
    field_size = (800, 600)
    env = environment.LTA_Discrete_ver0(step_time, field_size)
    env_params = {
            'agent_initial_position': np.array([40.,50.]),
            'agent_initial_velocity': np.array([50., 0.]),
            'agent_goal_position': np.array([300., 50.]),
            'agent_expected_speed': 50,
            'time_penalty_hyperparameter': 0.5,
            'max_ped_count': 0,
            'init_ped_count': 0,
            'add_ped_freq': 0,
            'rolling': True,
            'pixel2meters': 0.02,
            }
    env.set_params(env_params)
    env.display()
    env.reset()
    action = 15
    done = False
    while not done:
        obs, reward, done = env.step(action)
        show_observation(obs)
        print "reward: ", reward

def test4():
    print "==========TEST 4: INITIAL STEP WITHOUT DISPLAY==========="
    env = environment.LTA_Discrete_ver0(0.4, (800,600))
    env.set_params()
    obs = env.reset()
    show_observation(obs)
