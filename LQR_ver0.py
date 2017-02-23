import numpy as np
import pygame
import matplotlib.pyplot as plt
import scipy as sp

class Dynamic:
    def __init__(self, initial_state):
        """
        Initialize dynamic functions
        Input:
            initial_state: np1darray; size = 2, [x, v]
        """
        self._initial_state = np.asmatrix(initial_state).T
        self._step_time = 0.1
        self._time_counts = 100
        self.A = np.asmatrix(np.array([[1.,self._step_time],[0.,1.]]))
        self.B = np.asmatrix(np.array([0.5 * self._step_time**2, self._step_time])).T

        self.Q = np.asmatrix(np.array([[1.,0.],[0.,1.]]))
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
        record_F = []
        record_P = []
        record_P.append(self.H)

        for _ in range(self._time_counts):
            Pk = record_P[0]
            btp = np.dot(B.T, Pk)
            F = np.dot(np.dot(1 / (R + np.dot(btp, B)), btp), A)
            asbp = A - np.dot(B, F)
            P = Q + np.dot(np.dot(F.T, R), F) + np.dot(np.dot(asbp.T, Pk), asbp)
            record_F.insert(0, F)
            record_P.insert(0, P)
        
        actions = []
        states = [self._initial_state]
        total_cost = 0
        for t in range(self._time_counts):
            F = record_F[t]
            state = states[-1]
            action = -np.dot(F, states[-1])
            actions.append(action)
            next_state, cost = self._step(state, action)
            states.append(next_state)
            total_cost += cost
        total_cost += 0.5 * np.dot(states[-1].T, np.dot(self.H, states[-1]))
        print "total_cost: ", total_cost
        print "P cost: ", 0.5 * np.dot(states[0].T, np.dot(record_P[0], states[0]))
        return actions, states, total_cost


"""
init_state = np.array([200., -100.])
d = Dynamic(init_state) 
actions = []
for i in range(100):
    actions.append(np.asmatrix(0.))
states, cost = d.run(actions)
print "cost: ", cost
d.plot(states)
plt.show()
"""
init_state = np.array([200., -100.])
d = Dynamic(init_state) 
actions, states, cost = d.lqr_control()
d.plot(states)
plt.show()
