import tensorflow as tf

slim = tf.contrib.slim

class Critic(object):
    def __init__(self, sess, env_conf, train_conf):
        self._sess = sess

        # configuration of environment
        self._obs_shape = env_conf['obs_shape']
        self._act_dim = env_conf['act_dim']

        # configuration of training
        self._tau = train_conf['tau']
        self._lr = tf.placeholder(tf.float32, [], 'Critic_lr')

        # main network
        self._obs, self._act, self._out = self._create_critic_network('Critic')
        main_net = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Critic/')

        # target network
        self._obs_t, self._act_t, self._out_t, self._update_target = self._create_target_network()

        # placeholder for action-state value computed using bootstrapping target network
        self._g_from_target = tf.placeholder(tf.float32, [None,1], 'g_from_target')

        # gradient of action-state value function w.r.t. action
        self._act_grads = tf.gradients(self._out, self._act)[0]

        # optimization
        loss = tf.reduce_mean(tf.square(self._g_from_target-self._out))
        grads = tf.gradients(loss, main_net)
        grads_and_vars = zip(grads, main_net)
        opt = tf.train.AdamOptimizer(self._lr)
        self._train_op = opt.apply_gradients(grads_and_vars)

        # saver (restorer is the same)
        save_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Critic/') + \
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Critic_target/')
        self._saver = tf.train.Saver(save_vars, max_to_keep=3)

    def train(self, obs, act, g, lr):
        feed = {
            self._obs: obs,
            self._act: act,
            self._g_from_target: g,
            self._lr: lr
        }
        self._sess.run(self._train_op, feed_dict=feed)

    def infer(self, obs, act):
        feed = {
            self._obs: obs,
            self._act: act
        }
        out_ = self._sess.run(self._out, feed_dict=feed)
        return out_

    def target_infer(self, obs, act):
        feed = {
            self._obs_t: obs,
            self._act_t: act
        }
        out_ = self._sess.run(self._out_t, feed_dict=feed)
        return out_

    def update_target_network(self):
        self._sess.run(self._update_target)

    def action_gradients(self, obs, act):
        feed = {
            self._obs: obs,
            self._act: act
        }
        out_ = self._sess.run(self._act_grads, feed_dict=feed)
        return out_

    def save(self, save_path):
        self._saver.save(self._sess, save_path)
        print('Save Critic model to {}'.format(save_path))

    def load(self, load_path):
        self._saver.restore(self._sess, load_path)
        print('Load Critic model from {}'.format(load_path))

    def _create_critic_network(self, scope):
        with tf.variable_scope(scope):
            # input
            obs = tf.placeholder(tf.float32, [None, self._obs_shape[1], self._obs_shape[0], 3], 'obs_in')
            act = tf.placeholder(tf.float32, [None, self._act_dim], 'act_in')
            # network
            b_obs = slim.conv2d(obs, 32, [5,5], padding='VALID', scope='obs_conv1')
            b_obs = slim.max_pool2d(b_obs, [2,2], 2, scope='obs_pool1')
            b_obs = slim.conv2d(b_obs, 32, [5,5], padding='VALID', scope='obs_conv2')
            b_obs = slim.max_pool2d(b_obs, [2,2], 2, scope='obs_pool2')
            b_obs = slim.flatten(b_obs)
            b_obs = slim.fully_connected(b_obs, 400, scope='obs_fc3')

            b_act = slim.fully_connected(act, 400, scope='act_fc0')
            
            net = tf.concat(1, [b_obs, b_act])
            net = slim.fully_connected(net, 300, scope='fc1')
            out = slim.fully_connected(net, 1, activation_fn=None, scope='fc_out')

        return obs, act, out

    def _create_target_network(self):
        obs_t, act_t, out_t = self._create_critic_network('Critic_target')
        main_net = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Critic/')
        target_net = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Critic_target/')
        assert(len(main_net)==len(target_net))#DEBUG
        update_op = []
        for i in range(len(target_net)):
            new_target = tf.mul(main_net[i], self._tau) + tf.mul(target_net[i], (1.-self._tau))
            update_op.append(target_net[i].assign(new_target))

        return obs_t, act_t, out_t, update_op

