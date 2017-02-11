from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import import_env as my_env
from model.ddpg import DDPG

def main():
    gamma = 0.99
    tau = 0.001
    train_conf = {
        'n_episodes': 50000,
        'max_step': 200,
        'batch_size': 64,
        'buffer_size': 10000,
        'print_step': 100,
        'save_step': 1000,
        'actor_save_path': 'ckpt/actor',
        'critic_save_path': 'ckpt/critic',
        'actor_base_lr': 0.0001,
        'critic_base_lr': 0.001
    }

    # define environment
    env = my_env.make()

    # define DDPG model
    model = DDPG(env, gamma, tau)

    # train DDPG
    model.train(train_conf, display=True, verbose=False)

if __name__=='__main__':
    main()

