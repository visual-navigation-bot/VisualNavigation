import make
import numpy as np
import ctrl_LTA as ctrl

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


params = {
        'lambda1' : 2.33,
        'lambda2' : 2.073,
        'sigma_d' : 0.361,
        'sigma_w' : 2.088,
        'beta' : 1.462,
        'pixel2meters' : 0.02,
        'step_time': 0.4,
        'debug_mode': False
}
controller = ctrl.LTA_Controller(params)


env = make.make_environment('LTA_Continuous_ver0_Two_Peds_Walk')
episode_count = 2
sum_reward = 0.
for ep in range(episode_count):
    env.display()
    obs = env.reset()
    done = False
    total_reward = 0.
    while not done:
        action = controller.control(obs)
        obs, reward, done = env.step(action)
        total_reward += reward
    print "total reward: ", total_reward
    sum_reward += total_reward
print "==========TEST DONE=========="
print "LTA Controller average reward: ", sum_reward / episode_count

env = make.make_environment('LTA_Continuous_ver0_Two_Peds_Walk')
action = np.array([0.,0.])
episode_count = 2
sum_reward = 0.
for ep in range(episode_count):
    env.display()
    obs = env.reset()
    done = False
    total_reward = 0.
    while not done:
        obs, reward, done = env.step(action)
        total_reward += reward
    print "total reward: ", total_reward
    sum_reward += total_reward
print "==========TEST DONE=========="
print "No-op controller average reward: ", sum_reward / episode_count
