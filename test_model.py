import ctrl_iLQR_LTA as lta
import pygame
import numpy as np
import tensorflow as tf
import time
import random


def brute_force_calculate_energy(base, state, vstar):
    # brute force calculate energy by for loops
    # return I, S, D separately
    state_dims = state.shape[0]
    matS = []
    matD = []

    matk = []
    matq = []
    matt = []
    matd = []
    matE_v = []
    matwd = []
    matcos = []
    matwphi = []
    matE = []
    matI = []

    for ped_index in range(state_dims/4 - 1):
        ped = base._ped_list[ped_index]
        l1 = ped.lambda1
        l2 = ped.lambda2
        alpha = ped.alpha
        beta = ped.beta
        sd = ped.sigma_d
        sw = ped.sigma_w
        p2m = base.p2m
        
        positions = state[:state_dims/2]
        velocities = state[state_dims/2:]

        vt = velocities[ped_index*2+2: ped_index*2+4] * p2m
        p = positions[ped_index*2+2: ped_index*2+4] * p2m
        z = ped.goal_position * p2m
        u = ped.expected_speed * p2m
        v = vstar[ped_index*2:ped_index*2+2] *p2m

        S = l1 * (u - np.linalg.norm(v)) ** 2
        D = - l2 * np.dot((z-p), v) / (np.linalg.norm(v) * np.linalg.norm(z-p))
        matS.append(S)
        matD.append(D)


        k = np.array([]).reshape([0,2])
        q = np.array([]).reshape([0,2])
        t = np.array([]).reshape([0])
        d = np.array([]).reshape([0,2])
        E_v = np.array([]).reshape([0])
        wd = np.array([]).reshape([0])
        wphi = np.array([]).reshape([0])
        cos = np.array([]).reshape([0])
        E = np.array([]).reshape([0])

        #agent
        p2 = positions[:2] * p2m
        v2 = velocities[:2] * p2m
        k2 = p - p2
        q2 = v - v2
        t2 = max(-np.dot(k2, q2) / np.linalg.norm(q2) ** 2, 0)
        d2 = k2 + q2 * t2
        E_v2 = np.exp(- np.linalg.norm(d2)**2 / (2*sd**2))
        wd2 = np.exp(-np.linalg.norm(k2)**2 / (2*sw**2))
        print ""
        print "vdotk: ",np.dot(k2,vt)
        print "normk: ",np.linalg.norm(k2)
        print "vt: ", vt
        print ""
        cos2 = -np.dot(k2, vt) / (np.linalg.norm(k2) * np.linalg.norm(vt))
        wphi2 = ((1+cos2) / 2)**beta
        E2 = wphi2 * wd2 * E_v2

        k = np.vstack((k, k2.copy()))
        q = np.vstack((q, q2.copy()))
        t = np.hstack((t, t2))
        d = np.vstack((d, d2.copy()))
        E_v = np.hstack((E_v, E_v2))
        wd = np.hstack((wd, wd2))
        wphi = np.hstack((wphi, wphi2))
        cos = np.hstack((cos, cos2))
        E = np.hstack((E, E2))
        

        ped_count = state_dims / 4 - 1
        for i in range(ped_count):
            if i != ped_index:
                v2 = velocities[ped_index*2+2: ped_index*2+4] * p2m
                p2 = positions[ped_index*2+2: ped_index*2+4] * p2m
                k2 = p - p2
                q2 = v - v2
                t2 = max(-np.dot(k2, q2) / np.linalg.norm(q2) ** 2, 0)
                d2 = k2 + q2 * t2
                E_v2 = np.exp(- np.linalg.norm(d)**2 / (2*sd**2))
                wd2 = np.exp(-np.linalg.norm(k2)**2 / (2*sw**2))
                cos2 = -np.dot(k2, vt) / (np.linalg.norm(k2) * np.linalg.norm(vt))
                wphi2 = ((1+cos2) / 2)**beta
                E2 = wphi2 * wd2 * E_v2

                k = np.vstack((k, k2.copy()))
                q = np.vstack((q, q2.copy()))
                t = np.hstack((t, t2))
                d = np.vstack((d, d2.copy()))
                E_v = np.hstack((E_v, E_v2))
                wd = np.hstack((wd, wd2))
                wphi = np.hstack((wphi, wphi2))
                cos = np.hstack((cos, cos2))
                E = np.hstack((E, E2))
        I = np.sum(E)
        matk.append(np.expand_dims(k,0))
        matq.append(np.expand_dims(q,0))
        matt.append(np.expand_dims(t,0))
        matd.append(np.expand_dims(d,0))
        matE_v.append(np.expand_dims(E_v,0))
        matwd.append(np.expand_dims(wd,0))
        matwphi.append(np.expand_dims(wphi,0))
        matE.append(np.expand_dims(E,0))
        matI.append(I)
    print "speed energy: ", np.hstack(matS)
    print "direction energy: ", np.hstack(matD)
    print "interaction energy: ", np.hstack(matI)

    print "k: ", np.vstack(matk)
    print "q: ", np.vstack(matq)
    print "t: ", np.vstack(matt)
    print "d: ", np.vstack(matd)
    print "E_v: ", np.vstack(matE_v)
    print "wd: ", np.vstack(matwd)
    print "wphi: ", np.vstack(matwphi)
    print "cos: ", np.vstack(cos)


def test1():
    initial_state = np.array([100., 300.000001, 700., 300., 50., 0., -0., 50.])
    base = lta.LTA_Base(8)
    model = lta.LTA_Model(base)

    state_dims = initial_state.shape[0]
    ped_count = state_dims / 4 - 1
    ped_velocity = initial_state[state_dims/2+2:].reshape((ped_count,2))
    initial_vstar = (ped_velocity + np.random.normal(0, 2.5, size = [ped_count,2])).reshape(-1)
    print "initial state: ", initial_state
    print "initial vstar: ", initial_vstar

    brute_force_calculate_energy(base, initial_state, initial_vstar)

    feed = {model._initial_vstar: initial_vstar}
    model._sess.run(model._initialize_vstar_op, feed)

    feed = {model._state: initial_state}
    E, Ed, Es, Ei, k, q, t, d, E_v, wd, wphi, cos = model._sess.run([model._E_op, model._E_d_op, model._E_s_op, model._E_i_op, model.k, model.q, model.t, model.d, model.E_v, model.wd, model.wphi, model.cos] , feed)
    vdotk, normk, cp_vstar = model._sess.run([model.vdotk, model.normk, model.cp_vstar], feed)
    print ""
    print "vdotk: ", vdotk
    print "normk: ", normk
    print "cp_vstar: ", cp_vstar

    print ""
    print "ES: ", Es
    print "Ed: ", Ed
    print "Ei: ", Ei
    print "k: ", k
    print "q: ", q
    print "t: ", t
    print "d: ", d
    print "E_v: ", E_v
    print "wd: ", wd
    print "wphi: ", wphi
    print "cos: ", cos

def test2():
    initial_state = np.array([100., 300.000001, 700., 300., 50., 0., -0., 50.])
    base = lta.LTA_Base(8)
    model = lta.LTA_Model(base)

    state_dims = initial_state.shape[0]
    ped_count = state_dims / 4 - 1

    controller = lta.Initial_LTA_Controller(base, model)
    actions, states = controller.generate_steps(initial_state)
    t = 13
    state = states[t]
    action = actions[t]
    vstar = state[state_dims / 2+2:].copy()
    F = model.linearize_step(state,action,vstar)
    print F.shape
    print F
#test1()
test2()

    
