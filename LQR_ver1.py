import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

class Dynamic:
    def __init__(self, initial_state):
        """
        Initialize dynamic functions
        Input:
            initial_state: np1darray
        """
        self._initial_state = np.asmatrix(initial_state).T
        self._step_time = 0.1
        self._time_counts = 100
        self._state_dims = 2
        self._action_dims = 1
        self.A = np.asmatrix(np.array([[1.,self._step_time],[0.,1.]]))
        self.B = np.asmatrix(np.array([0.5 * self._step_time**2, self._step_time])).T

        self.Q = np.asmatrix(np.array([[0.5,0.],[0.,0.5]]))
        self.R = np.asmatrix(1.)#np.array([[1000., 1000.],[1000.,1.]])
        self.H = np.asmatrix(np.array([[1000., 0.],[0.,1000.]]))

    def run(self, actions):
        """
        step through the dynamics and solve return the corresponding states and cost
        Input:
            actions: list of npmatrix; 
        Return:
            states: list of npmatrix; including the final state
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

        cost += 0.5 * np.dot(states[-1].T, np.dot(self.H, states[-1]))
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
            state: npmatrix; the state vector
            action: npmatrix; the action vector
        Return:
            next_state: npmatrix; the next state vector
            cost: float; the step cost
        """
        next_state = np.dot(self.A, state) + np.dot(self.B, action)
        
        cost = 0.5 * (np.dot(state.T, np.dot(self.Q, state)) + np.dot(action.T, np.dot(self.R, action)))

        return next_state, cost

    def lqr_control(self):
        """
        Use the initial condition to form a lqr control and run it
        """
        # back prop lqr
        A = self.A
        B = self.B
        Q = self.Q
        H = self.H
        R = self.R

        K_record = []
        k_record = []
        V_record = [H]
        v_record = [np.asmatrix(np.zeros(self._state_dims)).T]
        
        for _ in range(self._time_counts):
            # calculate C, c, F, f
            F = np.hstack((A,B))
            f = np.asmatrix(np.zeros(self._state_dims)).T
            c = np.asmatrix(np.zeros(self._state_dims + self._action_dims)).T
            temp_zeros = np.asmatrix(np.zeros((self._state_dims, self._action_dims)))
            C = np.vstack((np.hstack((Q, temp_zeros)), np.hstack((temp_zeros.T, R))))

            K, k, V, v = self._get_K_V(F, f, C, c, V_record[0], v_record[0])
            K_record.insert(0, K)
            k_record.insert(0, k)
            V_record.insert(0, V)
            v_record.insert(0, v)
        actions = []
        states = [self._initial_state]
        total_cost = 0
        for t in range(self._time_counts):
            state = states[-1]
            action = np.dot(K_record[t], state) + k_record[t]
            actions.append(action)
            next_state, cost = self._step(state, action)
            states.append(next_state)
            total_cost += cost
        total_cost += 0.5 * np.dot(states[-1].T, np.dot(self.H, states[-1]))
        print "total_cost: ", total_cost
        print "P cost: ", 0.5 * np.dot(states[0].T, np.dot(V_record[0], states[0]))
        return states, total_cost


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




init_state = np.array([200., -100.])
d = Dynamic(init_state) 
states, cost = d.lqr_control()
d.plot(states)
plt.show()
