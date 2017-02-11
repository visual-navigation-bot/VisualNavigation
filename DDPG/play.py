from __future__ import print_function
from __future__ import division

import numpy as np
import gym

from model.ddpg import DDPG

# use trained DDPG to play game
def main():
    gamma = 0.99
    tau = 0.001
    n_episodes = 50
    max_step = 100
    actor_load_path = './ckpt/actor20170210-174344.ckpt'
    critic_load_path = './ckpt/critic20170210-174344.ckpt'

    # define environment
    env = gym.make('Pendulum-v0')

    # define DDPG model
    policy = DDPG(env, gamma, tau)
    policy.load_model(actor_load_path, critic_load_path)

    for _ in range(n_episodes):
        g = policy.infer(max_step)

        print('Expected total return = {}'.format(g))

if __name__=='__main__':
    main()

