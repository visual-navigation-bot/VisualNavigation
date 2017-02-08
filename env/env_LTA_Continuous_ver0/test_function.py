import environment
import numpy as np

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


def test1():
    print "==========TEST 1: INITIAL STEP WITH DISPLAY==========="
    env = environment.LTA_Continuous_ver0(0.4, (800,600))
    env.set_params()
    env.display()
    obs = env.reset()
    show_observation(obs)

def test2():
    print "==========TEST 2: INITIAL STEP WITHOUT DISPLAY==========="
    env = environment.LTA_Continuous_ver0(0.4, (800,600))
    env.set_params()
    obs = env.reset()
    show_observation(obs)

def test3():
    print "==========TEST 3: NO OP WITH DISPLAY==========="
    env = environment.LTA_Continuous_ver0(0.4, (800,600))
    env.set_params()
    obs = env.reset()
    show_observation(obs)

    action = np.array([0.,0.])
    obs, r, done = env.step(action)
    show_observation(obs)
    print "reward: ",r
    print "done: ", done

def test4():
    print "==========TEST 4: CONTINUOUS NO OP WITH DISPLAY==========="
    env = environment.LTA_Continuous_ver0(0.4, (800,600))
    env.set_params()
    env.display()
    obs = env.reset()
    show_observation(obs)

    done  = False
    action = np.array([0.,0.])
    while not done:
        obs, r, done = env.step(action)
        print "reward: ", r







