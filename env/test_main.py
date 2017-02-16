import make
import numpy as np

"""
env = make.make_environment('LTA_Continuous_ver0_Opposite_Walk')
action = np.array([0.,0.])

env.display()
env.reset()
done = False
while not done:
    obs, reward, done = env.step(action)
    print "reward: ", reward

env = make.make_environment('LTA_Discrete_ver0_Opposite_Walk')

env.display()
env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done = env.step(action)
    print "reward: ", reward
"""

env = make.make_environment('LTA_Continuous_ver0_Two_Peds_Walk')
action = np.array([0.,0.])

#env.display()
env.reset()
done = False
total_reward = 0.
while not done:
    obs, reward, done = env.step(action)
    total_reward += reward
    print "reward: ", reward

print "total reward: ", total_reward

