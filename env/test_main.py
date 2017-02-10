import make
import numpy as np

env = make.make_environment('LTA_Continuous_ver0_Opposite_Walk')
action = np.array([0.,0.])

env.display()
env.reset()
done = False
while not done:
    obs, reward, done = env.step(action)
    print "reward: ", reward
