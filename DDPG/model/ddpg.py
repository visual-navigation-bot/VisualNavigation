import tensorflow as tf
import numpy as np
from datetime import datetime
from scipy.signal import lfilter
import scipy.misc as misc
import sys#DEBUG

from critic import Critic
from actor import Actor
from replay_buffer import ReplayBuffer
from ou_noise import OU_Noise

class DDPG(object):
    def __init__(self, env, gamma, tau):
        self._env = env
        self._gamma = gamma
        self._tau = tau

        # obtain configuration of action and observation space
        # --> to define I/O of actor and critic
        act_space = self._env.action_space
        obs_space = self._env.observation_space

        self._rs_obs_shape = rs_obs_shape = (64,64) #obs_space.field_size
        act_low = act_space.low
        act_high = act_space.high
        act_dim = len(act_low)

        # define session
        sess_conf = tf.ConfigProto()
        sess_conf.gpu_options.allow_growth = True
        sess_conf.gpu_options.per_process_gpu_memory_fraction = 0.4
        self._sess = tf.Session(config=sess_conf)

        # define actor
        env_conf = {
            'obs_shape': rs_obs_shape,
            'act_dim': act_dim,
            'act_low': act_low,
            'act_high': act_high
        }
        train_conf = {'tau': tau}
        self._actor = Actor(self._sess, env_conf, train_conf)

        # define critic
        env_conf = {
            'obs_shape': rs_obs_shape,
            'act_dim': act_dim
        }
        train_conf = {'tau': tau}
        self._critic = Critic(self._sess, env_conf, train_conf)

        # exploration noise
        self._noise = OU_Noise(env_conf['act_dim'])

    def train(self, conf, display=False, verbose=False):
        # training configuration
        n_episodes = conf['n_episodes']
        max_step = conf['max_step']
        batch_size = conf['batch_size']
        print_step = conf['print_step']
        save_step = conf['save_step']
        actor_save_path = conf['actor_save_path'] + datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S') + '.ckpt'
        critic_save_path = conf['critic_save_path'] + datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S') + '.ckpt'

        # replay buffer configuration
        buffer_size = conf['buffer_size']
        min_buffer_size = int(conf['buffer_size'] / 100)
        replay_buffer = ReplayBuffer(buffer_size)

        # learning rate
        actor_lr = conf['actor_base_lr']
        critic_lr = conf['critic_base_lr']

        # initialize model
        init_op = tf.global_variables_initializer()
        self._sess.run(init_op)

        if display:
            self._env.display()

        global_step = 0
        for episode in range(n_episodes):
            # reset and start an new episode
            o = self._env.reset()
            o = self._resize_obs(o) # resize to a smaller image
            self._noise.reset()

            for step in range(max_step):
                # interact with to environment and save to experience replay
                a = self._actor.infer(o)
                a += self._noise.noise() # add OU noise for exploration
                o2, r, terminal = self._env.step(np.reshape(a, (-1,)))
                o2 = self._resize_obs(o2) # resize to a smaller image

                replay_buffer.add(o, a, r, terminal, o2)#TODO: check shape of added data

                print(step)

                # sample a batch of data from replay buffer and train
                if replay_buffer.size > min_buffer_size:
                    data = replay_buffer.sample_batch(batch_size)
                    # (observation_t, action_t, expected_total_return_t, terminal_t, observation_(t+1))
                    # Note that expected total return (g) is computed based on critic's target network
                    o_batch, a_batch, g_batch, o2_batch = self._preprocess_data(data)

                    # update critic
                    self._critic.train(o_batch, a_batch, g_batch, critic_lr)

                    # update actor (a_batch_now is similar to argmax_a(Q(s,a)) in Q learning,
                    # estimated actions, whereas a_batch are sampled actions with exploration noise)
                    a_batch_est = self._actor.infer(o_batch)
                    act_grads = self._critic.action_gradients(o_batch, a_batch_est) / batch_size #normalize over batch
                    self._actor.train(o_batch, act_grads, actor_lr)

                    # update target network
                    self._actor.update_target_network()
                    self._critic.update_target_network()

                    # print something
                    if global_step%print_step==0:
                        predicted_q = self._critic.infer(o_batch, a_batch_est)
                        print('{}/{} episodes, {}/{} steps: Q_mean = {}' \
                            .format(episode, n_episodes, step, max_step, np.mean(predicted_q)))

                    # save checkpoint
                    if (global_step%save_step==0) and global_step>0:
                        self.save_model(actor_save_path, critic_save_path)
                        print('Save model to {}(actor), {}(critic)'.format(actor_save_path, critic_save_path))

                    global_step += 1

                # update observation
                o = o2

                # check if episode ends
                if terminal:
                    if verbose:
                        print('Episode ends at {}/{}'.format(step+1,max_step))
                    break

        print('End training. Elapsed iteration = {}'.format(global_step))

    def infer(self, max_step, display=True):
        o = self._env.reset()
        r_list = []
        if display:
            self._env.display()
        for _ in range(max_step):
            # agent takes an action and interacts with environment
            a = self._actor.infer(o.reshape((1,3)))
            o2, r, t, _ = self._env.step(a)

            # update observation
            o = o2

            # record immediate return
            r_list.append(r)

        # compute expected total reward
        g = self._compute_total_return(r_list)

        return g

    def load_model(self, actor_path, critic_path):
        self._actor.load(actor_path)
        self._critic.load(critic_path)

    def save_model(self, actor_path, critic_path):
        self._actor.save(actor_path)
        self._critic.save(critic_path)

    def _preprocess_data(self, data):
        ### use target network to compute expected total return
        # extract from tuple
        batch_size = len(data)
        o_batch = np.vstack([_[0] for _ in data])
        a_batch = np.vstack([_[1] for _ in data])
        r_batch = np.vstack([_[2] for _ in data])
        t_batch = np.vstack([_[3] for _ in data])
        o2_batch = np.vstack([_[4] for _ in data])
       
        # evaluate boostrapped next-step action-state value using target network
        a2_batch = self._actor.target_infer(o2_batch)
        o2a2_value = self._critic.target_infer(o2_batch, a2_batch) 
        
        # compute expected total return
        g_batch = []
        for i in range(batch_size):
            if t_batch[i]:
                # episode terminates
                g_batch.append(r_batch[i])
            else:
                g_batch.append(r_batch[i] + self._gamma*o2a2_value[i])
        g_batch = np.array(g_batch).reshape((batch_size,1))

        return o_batch, a_batch, g_batch, o2_batch

    def _compute_total_return(self, r_list):
        x = np.array(r_list)
        g_all = lfilter([1], [1,-self._gamma], x[::-1])[::-1]
        
        return g_all[-1][0]

    def _resize_obs(self, o):
        o = misc.imresize(o, self._rs_obs_shape)
        o = np.expand_dims(o, axis=0)

        return o
