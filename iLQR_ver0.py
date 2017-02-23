import numpy as np
import pygame
import matplotlib.pyplot as plt
import scipy as sp
import tensorflow as tf
import time

class Model:
    def __init__(self, initial_state):
        """
        Given Initial State, initialize the dynamic model
        Input: 
            initial_state: np1darray; the initial state of the model
        """
        self._initial_state = np.asmatrix(initial_state)
        self._state_dims = 2
        self._action_dims = 1
        self._b = 0.0005
        self._step_time = 0.1
        self._s = tf.placeholder(tf.float32, [self._state_dims])
        self._a = tf.placeholder(tf.float32, [self._action_dims])
        with tf.name_scope('step'):
            v = self._s[1] - self._b * tf.pow(self._s[1], 2) * self._step_time + self._a[0] * self._step_time
            x = self._s[0] + (self._s[1] + v) * self._step_time / 2
            self._next_s = tf.pack([x,v])
            self._cost = 0.5 * tf.pow(tf.abs(self._s[1]), 3) + 0.5 * tf.pow(tf.abs(self._s[0]), 3) + 0.00001 * tf.pow(tf.abs(self._a[0]), 2)

            self._f = self._next_s
            F = []
            for index_s in range(self._state_dims):
                F.append(tf.concat(0, [tf.gradients(self._next_s[index_s], self._s)[0], tf.gradients(self._next_s[index_s], self._a)[0]]))
            self._F = tf.pack(F)

            Qxx = []
            self._qx = tf.gradients(self._cost, self._s)[0]
            for index_s in range(self._state_dims):
                Qxx.append(tf.gradients(self._qx[index_s], self._s)[0])
            self._Qxx = tf.pack(Qxx)

            Quu = []
            self._qu = tf.gradients(self._cost, self._a)[0]
            for index_a in range(self._action_dims):
                Quu.append(tf.gradients(self._qu[index_a], self._a)[0])
            self._Quu = tf.pack(Quu)

        with tf.name_scope('final'):
            self._final_cost = 1000. * tf.pow(tf.abs(self._s[1]), 3) + 1000. * tf.pow(tf.abs(self._s[0]), 3)

            self._vx = tf.gradients(self._final_cost, self._s)[0]
            Vxx = []
            for index_s in range(self._state_dims):
                Vxx.append(tf.gradients(self._vx[index_s], self._s)[0])
            self._Vxx = tf.pack(Vxx)

        self._sess = tf.Session()
        self._init = tf.initialize_all_variables()
        self._sess.run(self._init)

    def step(self, state, action):
        """
        Given state and action, return next state and step cost
        Input:
            state: np1darray; the state
            action: np1darray; the action
        Output:
            next_state: np1darray; the next state
            cost: float; the cost
        """
        feed = {self._s: state, self._a: action}
        s, c = self._sess.run([self._next_s, self._cost], feed)
        return s, c

    def linearize_step(self, state, action):
        """
        linearize the model on a specific state and action
        Input:
            state: np1darray; the state
            action: np1darray; the action
        Return:
            list of nparray:
                F: np2darray; nx = Fdx + f
                f: np1darray;
                Qxx: np2darray; hessian of cost to x
                Quu: np2darray; hessian of cost to u
                qx: np1darray; gradient of cost to x
                qu: np1darray; gradient of cost to u
        """
        feed = {self._s: state, self._a: action}
        linear_params = self._sess.run([self._F, self._f, self._Qxx, self._Quu, self._qx, self._qu], feed)
        return linear_params

    def final_cost(self, final_state):
        """
        return the final cost by given final state
        Input: 
            final_state: np1darray; the final state
        Return: 
            final_cost: float; the final cost
        """
        feed = {self._s: final_state}
        c = self._sess.run(self._final_cost, feed)
        return c

    def linearize_final_cost(self, final_state):
        """
        return the linearized final cost
        Input:
            final_state: np1darray; the final state
        Return:
            list of nparray:
                Vxx: np2darray; hessian of cost to xf
                vx: np1darray; gradient of cost to xf
        """
        feed = {self._s: final_state}
        linear_params = self._sess.run([self._Vxx, self._vx], feed)
        return linear_params


class LQR_Optimizer:
    def __init__(self, initial_state, actions, model, states = None):
        """
        Initialize dynamic functions
        Input:
            initial_state: np1darray
            actions: a list of np1darray
        """
        self._initial_state = initial_state
        self._step_time = 0.1
        self._time_counts = 100
        self._state_dims = 2
        self._action_dims = 1
        self._model = model
        self._tracked_actions = actions
        if states is None:
            self._tracked_states, _ = self.run(actions)
        else:
            self._tracked_states = states

    def run(self, actions):
        """
        step through the dynamics and solve return the corresponding states and cost
        Input:
            actions: list of nparray; 
        Return:
            states: list of nparray; including the final state
            cost: float; the total cost
        """
        assert len(actions) == self._time_counts, "not enough action pairs"
        states = []
        cost = 0
        states.append(self._initial_state.copy())
        for action in actions:
            next_state, step_cost = self._step(states[-1], action)
            states.append(next_state.copy())
            cost += step_cost

        cost += self._model.final_cost(states[-1])
        print "initialize cost: ", cost
        return states, cost

    def plot(self, states):
        time = np.arange(self._time_counts + 1)
        x = []
        for state in states:
            x.append(state.tolist()[0])
        plt.figure(1)
        plt.plot(time, x)

    def _step(self, state, action):
        """
        Take one step
        Input:
            state: nparray; the state vector
            action: nparray; the action vector
        Return:
            next_state: nparray; the next state vector
            cost: float; the step cost
        """
        next_state, cost = self._model.step(state, action)

        return next_state, cost

    def lqr_control(self):
        """
        Use the initial condition to form a lqr control and run it
        """

        final_state = self._tracked_states[-1]
        V, v = self._model.linearize_final_cost(final_state)
        # back prop lqr
        K_record = []
        k_record = []
        V_record = [np.asmatrix(V)]
        v_record = [np.asmatrix(v).T]
        
        for t in range(self._time_counts - 1, -1, -1):
            # calculate C, c, F, f
            tracked_state = self._tracked_states[t]
            tracked_action = self._tracked_actions[t]
            F, f, Cxx, Cuu, cx, cu = self._model.linearize_step(tracked_state, tracked_action)
            mat_F = np.asmatrix(F)
            #mat_f = np.asmatrix(f).T
            mat_f = np.asmatrix(np.zeros(self._state_dims)).T
            mat_Cxx = np.asmatrix(Cxx)
            mat_Cuu = np.asmatrix(Cuu)
            mat_cx = np.asmatrix(cx).T
            mat_cu = np.asmatrix(cu).T
            """
            print mat_F
            print mat_f
            print mat_Cxx
            print mat_Cuu
            print mat_cx
            print mat_cu
            print ""
            """
            temp_zeros = np.asmatrix(np.zeros((self._state_dims, self._action_dims)))
            mat_C = np.vstack((np.hstack((mat_Cxx, temp_zeros)), np.hstack((temp_zeros.T, mat_Cuu))))
            mat_c = np.vstack((mat_cx, mat_cu))

            K, k, V, v = self._get_K_V(mat_F, mat_f, mat_C, mat_c, V_record[0], v_record[0])
            K_record.insert(0, K)
            k_record.insert(0, k)
            V_record.insert(0, V)
            v_record.insert(0, v)
        actions = []
        states = [self._initial_state]
        total_cost = 0
        for t in range(self._time_counts):
            state = np.asmatrix(states[-1]).T
            tracked_state = np.asmatrix(self._tracked_states[t]).T
            tracked_action = np.asmatrix(self._tracked_actions[t]).T
            action = np.asarray(np.dot(K_record[t], state - tracked_state) + k_record[t] + tracked_action).reshape(-1)
            actions.append(action)
            next_state, cost = self._step(states[-1], action)
            states.append(next_state)
            total_cost += cost
        cost += self._model.final_cost(states[-1])
        print "total_cost: ", total_cost
        return actions, states


    def _get_K_V(self, F, f, C, c, V, v):
        ftv = np.dot(F.T, V)
        Q = C + np.dot(ftv, F)
        q = c + np.dot(ftv, f) + np.dot(F.T, v)
        Quu = Q[self._state_dims:, self._state_dims:]
        Qxx = Q[:self._state_dims, :self._state_dims]
        Qxu = Q[:self._state_dims, self._state_dims:]
        Qux = Q[self._state_dims:, :self._state_dims]
        qx = q[:self._state_dims,:]
        qu = q[self._state_dims:, :]
        invQuu = np.linalg.inv(Quu)
        K = -np.dot(invQuu, Qux)
        k = -np.dot(invQuu, qu)
        V = Qxx + np.dot(Qxu, K) + np.dot(K.T, Qux) + np.dot(np.dot(K.T, Quu), K)
        v = qx + np.dot(Qxu, k) + np.dot(K.T, qu) + np.dot(np.dot(K.T, Quu), k)
        return K, k, V, v


init_state = np.array([-200, 100])
actions = [np.zeros(1) for _ in range(100)]
states = None
m = Model(init_state)
for _ in range(10):
    d = LQR_Optimizer(init_state, actions, m, states) 
    actions, states = d.lqr_control()
    d.plot(states)
plt.show()
