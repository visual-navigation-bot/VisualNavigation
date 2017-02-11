import tensorflow as tf
import sys#DEBUG

slim = tf.contrib.slim

class Actor(object):
    def __init__(self, sess, env_conf, train_conf):
        self._sess = sess
        
        # configuration of environment
        self._obs_shape = env_conf['obs_shape']
        self._act_dim = env_conf['act_dim']
        self._act_low = env_conf['act_low']
        self._act_high = env_conf['act_high']

        # configuration of training
        self._tau = train_conf['tau']
        self._lr = tf.placeholder(tf.float32, [], 'Actor_lr')

        # main network
        self._obs, self._out, self._final_out = self._create_actor_network('Actor')
        main_net = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Actor/')

        # target network
        self._obs_t, self._final_out_t, self._update_target = self._create_target_network()

        # placeholder for actor gradient
        self._act_grads = tf.placeholder(tf.float32, [None,self._act_dim], 'action_grads')

        # optimization
        grads = tf.gradients(self._final_out, main_net, -self._act_grads)
        grads_and_vars = zip(grads, main_net)
        opt = tf.train.AdamOptimizer(self._lr)
        self._train_op = opt.apply_gradients(grads_and_vars)

        # saver (restorer is the same as saver)
        save_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Actor/') + \
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Actor_target/')
        self._saver = tf.train.Saver(save_vars, max_to_keep=3)

    def train(self, obs, act_grads, lr):
        feed = {
            self._obs: obs,
            self._act_grads: act_grads,
            self._lr: lr
        }
        self._sess.run(self._train_op, feed_dict=feed)

    def infer(self, obs):
        feed = {self._obs: obs}
        out_ = self._sess.run(self._final_out, feed_dict=feed)
        return out_

    def target_infer(self, obs):
        feed = {self._obs_t: obs}
        out_ = self._sess.run(self._final_out_t, feed_dict=feed)
        return out_

    def update_target_network(self):
        self._sess.run(self._update_target)

    def save(self, save_path):
        self._saver.save(self._sess, save_path)
        print('Save Actor model to {}'.format(save_path))

    def load(self, load_path):
        self._saver.restore(self._sess, load_path)
        print('Load Actor model from {}'.format(load_path))

    def _create_actor_network(self, scope):
        with tf.variable_scope(scope):
            # input
            obs = tf.placeholder(tf.float32, [None, self._obs_shape[1], self._obs_shape[0], 3], 'obs_in')
            # network
            net = slim.conv2d(obs, 32, [5,5], padding='VALID', scope='conv1')
            net = slim.max_pool2d(net, [2,2], 2, scope='pool1')
            net = slim.conv2d(net, 32, [5,5], padding='VALID', scope='conv2')
            net = slim.max_pool2d(net, [2,2], 2, scope='pool2')
            net = slim.flatten(net)
            net = slim.fully_connected(net, 400, scope='fc3')
            net = slim.fully_connected(net, 300, scope='fc4')
            assert(self._act_dim==2)
            out1 = slim.fully_connected(net, 1, activation_fn=tf.nn.sigmoid, scope='fc_out1')
            out2 = slim.fully_connected(net, 1, activation_fn=tf.nn.tanh, scope='fc_out2')
            out = tf.concat(1, [out1, out2])
            # scale to action range
            final_out = tf.mul(out, self._act_high)

        return obs, out, final_out

    def _create_target_network(self):
        obs_t, _, final_out_t = self._create_actor_network('Actor_target')
        main_net = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Actor/')
        target_net = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Actor_target/')
        assert(len(main_net)==len(target_net))#DEBUG
        update_op = []
        for i in range(len(target_net)):
            new_target = tf.mul(main_net[i], self._tau) + tf.mul(target_net[i], (1.-self._tau))
            update_op.append(target_net[i].assign(new_target))

        return obs_t, final_out_t, update_op

