import tensorflow as tf
import numpy as np
import random
import time
import pygame

def display_states(base, states):
    pygame.init()
    screen = pygame.display.set_mode((800,600))
    pygame.display.set_caption('LTA_iLQR')
    screen.fill((255,255,255))
    clock = pygame.time.Clock()
    running = True
    state_index = 0
    while running:
        clock.tick(base.frame_per_second)
        screen.fill((255,255,255))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        state = states[state_index]

        state_dims = state.shape[0]
        positions = state[:state_dims / 2]
        velocities = state[state_dims / 2:]

        agent_position = positions[:2]
        agent_display_position = (int(agent_position[0]), int(agent_position[1]))
        agent_velocity = velocities[:2]
        end_position = agent_velocity * base.step_time + agent_position
        end_display_position = (int(end_position[0]), int(end_position[1]))
        pygame.draw.circle(screen, (255,0,0), agent_display_position, 3, 0)
        pygame.draw.line(screen, (0,0,0), agent_display_position, end_display_position, 1)

        goal_position = base.agent.goal_position
        goal_display_position = (int(goal_position[0]), int(goal_position[1]))
        pygame.draw.circle(screen, (0,0,255), goal_display_position, 6, 0)

        ped_count = state_dims / 4 - 1
        end_positions = positions + velocities * base.step_time
        for ped_index in range(ped_count):
            position = positions[ped_index * 2 + 2: ped_index * 2 + 4]
            end_position = end_positions[ped_index*2 + 2: ped_index * 2 + 4]
            display_position = (int(position[0]), int(position[1]))
            pygame.draw.circle(screen, (0,0,0), display_position, 3, 0)
            end_display_position = (int(end_position[0]), int(end_position[1]))
            pygame.draw.line(screen, (0,0,0), display_position, end_display_position, 1)
        pygame.display.flip()


        state_index += 1
        if state_index >= len(states):
            running = False



def show_observation(observation):
    print "#####Observation#####"
    print "AGENT INFO:"
    print "    position: ",observation['agent_position']
    print "    velocity: ",observation['agent_velocity']
    print "    goal position: ", observation['agent_goal_position']
    print "PEDESTRIAN INFO:"
    for ped_index in range(len(observation['ped_ID'])):
        print "    ID: ", observation['ped_ID'][ped_index]
        print "    position: ", observation['ped_position'][ped_index]
        print "    velocity: ", observation['ped_velocity'][ped_index]
        print ""
    print "#####################"

def show_states(states):
    print "######SHOW STATES######"
    t = 0
    for state in states:
        print "t = ", t
        print state[0:len(state) / 2]
        print state[len(state) / 2: len(state)]
        t += 1

def show_actions(actions):
    print "#####SHOW ACTIONS#######"
    t = 0
    for action in actions:
        print "t = ", t
        print action[0: len(action) / 2]
        print action[len(action) / 2: len(action)]
        t += 1

class Pedestrian_Base:
    def __init__(self, ped_params):
        self._lambda1 = ped_params['lambda1']
        self._lambda2 = ped_params['lambda2']
        self._sigma_d = ped_params['sigma_d']
        self._sigma_w = ped_params['sigma_w']
        self._beta = ped_params['beta']
        self._alpha = ped_params['alpha']
        self._expected_speed = ped_params['expected_speed']
        self._goal_position = ped_params['goal_position']
    @property
    def lambda1(self):
        return self._lambda1
    @property
    def lambda2(self):
        return self._lambda2
    @property
    def sigma_d(self):
        return self._sigma_d
    @property
    def sigma_w(self):
        return self._sigma_w
    @property
    def beta(self):
        return self._beta
    @property
    def alpha(self):
        return self._alpha
    @property
    def expected_speed(self):
        return self._expected_speed
    @property
    def goal_position(self):
        return self._goal_position

class Agent_Base:
    def __init__(self, agent_params):
        self._lambda1 = agent_params['lambda1']
        self._lambda2 = agent_params['lambda2']
        self._sigma_d = agent_params['sigma_d']
        self._sigma_w = agent_params['sigma_w']
        self._beta = agent_params['beta']
        self._expected_speed = agent_params['expected_speed']
        self._goal_position = agent_params['goal_position']
    @property
    def lambda1(self):
        return self._lambda1
    @property
    def lambda2(self):
        return self._lambda2
    @property
    def sigma_d(self):
        return self._sigma_d
    @property
    def sigma_w(self):
        return self._sigma_w
    @property
    def beta(self):
        return self._beta
    @property
    def expected_speed(self):
        return self._expected_speed
    @property
    def goal_position(self):
        return self._goal_position



        
class LTA_Base:
    """
    The base of LTA model
    No matter what initial state it is, the base is always the same
    """
    # including pedestrian information and time counts
    def __init__(self, state_dims, agent_params, ped_params_list):
        self._step_time = 0.4
        self._frame_per_second = 1. / self._step_time
        self._p2m = 0.02
        self._ped_list = []
        for ped_index in range(len(ped_params_list)):
            ped_params = ped_params_list[ped_index]
            self._ped_list.append(Pedestrian_Base(ped_params))
        self._agent = Agent_Base(agent_params)
        self._lambda1 = np.hstack([ped.lambda1 for ped in self._ped_list])
        self._lambda2 = np.hstack([ped.lambda2 for ped in self._ped_list])
        self._sigma_d = np.hstack([ped.sigma_d for ped in self._ped_list])
        self._sigma_w = np.hstack([ped.sigma_w for ped in self._ped_list])
        self._beta = np.hstack([ped.beta for ped in self._ped_list])
        self._alpha = np.hstack([ped.alpha for ped in self._ped_list])
        self._ped_expected_speed = np.hstack([ped.expected_speed for ped in self._ped_list])
        self._ped_goal_position = np.vstack([ped.goal_position for ped in self._ped_list])

        self._state_dims = state_dims


    @property
    def step_time(self):
        return self._step_time
    @property
    def frame_per_second(self):
        return self._frame_per_second
    @property
    def p2m(self):
        return self._p2m
    @property
    def agent(self):
        return self._agent
    @property
    def lambda1(self):
        return self._lambda1
    @property
    def lambda2(self):
        return self._lambda2
    @property
    def sigma_d(self):
        return self._sigma_d
    @property
    def sigma_w(self):
        return self._sigma_w
    @property
    def beta(self):
        return self._beta
    @property
    def alpha(self):
        return self._alpha
    @property
    def ped_expected_speed(self):
        return self._ped_expected_speed
    @property
    def ped_goal_position(self):
        return self._ped_goal_position
    @property
    def state_dims(self):
        return self._state_dims


class LTA_Model:
    def __init__(self, base):
        self._state_dims = base.state_dims
        self._action_dims = 2
        self._step_time = 0.4
        self._base = base
        self._state = tf.placeholder(tf.float64, [self._state_dims])
        self._action = tf.placeholder(tf.float64, [self._action_dims])

        self._initial_vstar = tf.placeholder(tf.float64, [self._state_dims / 2 - 2])
        self._vstar = tf.Variable(tf.zeros([self._state_dims / 2 - 2], dtype = tf.float64), name = 'vstar')
        self._initialize_vstar_op = tf.assign(self._vstar, self._initial_vstar)

        p2m = tf.constant(self._base.p2m, dtype = tf.float64)
        l1 = tf.constant(self._base.lambda1, dtype = tf.float64)
        l2 = tf.constant(self._base.lambda2, dtype = tf.float64)
        sd = tf.constant(self._base.sigma_d, dtype = tf.float64)
        sw = tf.constant(self._base.sigma_w, dtype = tf.float64)
        b = tf.constant(self._base.beta, dtype = tf.float64)
        a = tf.constant(self._base.alpha, dtype = tf.float64)



        with tf.name_scope('energy_func'):
            positions = tf.reshape(self._state[:self._state_dims / 2], [self._state_dims / 4, 2])
            velocities = tf.reshape(self._state[self._state_dims / 2:], [self._state_dims / 4, 2])
            ped_position = tf.reshape(self._state[2:self._state_dims / 2], [self._state_dims / 4 - 1, 2])
            ped_velocity = tf.reshape(self._state[self._state_dims / 2 + 2:], [self._state_dims / 4 - 1, 2])
            vstar = tf.reshape(self._vstar, [self._state_dims / 4 - 1, 2])

            ped_expected_speed = tf.constant(self._base.ped_expected_speed, dtype = tf.float64)
            ped_goal_position = tf.constant(self._base.ped_goal_position, dtype = tf.float64)

            # E_i = sigma(i)(wr(i) * exp(- d**2 / (2**sd**2)))
            # q = v - vj; k = pi - pj; cos(phi) = -kdotvt / (|k|*|vt|)
            # i is this pedestrian
            # d = k - kdotq * q / |q|**2
            # wr = exp(-k ** 2 / (2 * sw**2)) * ( (1+cos(phi)) / 2)**beta

            v = p2m * vstar
            vt = p2m * ped_velocity
            u = p2m * ped_expected_speed
            z = p2m * ped_goal_position
            p = p2m * ped_position

            normv = tf.sqrt(tf.reduce_sum(tf.square(v), 1))
            E_s = l1 * tf.square(u - normv)
            #################
            self._E_s_op = E_s
            #################

            norml = tf.sqrt(tf.reduce_sum(tf.square(z - p), 1))
            pdotv = tf.reduce_sum(v * (z-p), 1)
            E_d = - l2 * (pdotv / (norml * normv))

            #################
            self._E_d_op = E_d
            #################

            ped_count = self._state_dims / 4 - 1
            tiled_position = tf.reshape(tf.tile(positions, [ped_count+1, 1]), [ped_count+1, ped_count + 1, 2])
            diff_position = tf.transpose(tiled_position, [1, 0, 2]) - tiled_position
            unpacked_diff_position = tf.reshape(diff_position, [(ped_count + 1)*(ped_count + 1) * 2])
            
            #indexing, using position indexing
            diff_position_index = np.arange(0, (ped_count + 1)*(ped_count + 1) * 2).reshape(ped_count+1, ped_count+1, 2)
            diff_position_index = np.delete(diff_position_index, 0, 0)
            del_index = np.arange(ped_count) + 1
            mask = np.ones((ped_count, ped_count + 1), dtype = np.bool)
            mask[range(ped_count), del_index] = False
            diff_position_index = tf.constant(diff_position_index[mask].reshape(-1))

            k = tf.reshape(tf.gather(unpacked_diff_position, diff_position_index), [ped_count, ped_count, 2]) * p2m
            #################
            self.k = k
            #################


            tiled_vstar = tf.transpose(tf.reshape(tf.tile(vstar, [ped_count+1,1]), [ped_count+1, ped_count,2]), [1,0,2])
            tiled_velocity = tf.reshape(tf.tile(velocities, [ped_count, 1]), [ped_count, ped_count+1, 2])
            unpacked_diff_velocity = tf.reshape(tiled_vstar - tiled_velocity, [ped_count*(ped_count + 1)*2])

            #indexing, using velocity indexing
            diff_velocity_index = np.arange(0, ped_count*(ped_count+1)*2).reshape(ped_count, ped_count+ 1, 2)
            diff_velocity_index = tf.constant(diff_velocity_index[mask].reshape(-1))

            q = tf.reshape(tf.gather(unpacked_diff_velocity, diff_velocity_index), [ped_count, ped_count, 2]) * p2m
            #################
            self.q = q
            #################


            kdotq = tf.reduce_sum(k * q, 2)
            normqsq = tf.reduce_sum(tf.square(q), 2)
            t = tf.nn.relu(- kdotq / normqsq)
            d = k + q*tf.reshape(t, [ped_count, ped_count, 1])
            #################
            self.t = t
            self.d = d
            #################
            normdsq = tf.reduce_sum(tf.square(d), 2)
            E_v = tf.exp(- normdsq / tf.reshape(2 * tf.square(sd), [ped_count, 1]))
            #################
            self.E_v = E_v
            #################

            normk = tf.sqrt(tf.reduce_sum(tf.square(k), 2))
            wd = tf.exp(- tf.square(normk) / tf.reshape(2 * tf.square(sw), [ped_count, 1]))
            copied_vt = tf.transpose(tf.reshape(tf.tile(vt, [ped_count,1]),[ped_count,ped_count,2]),[1,0,2])
            normvt = tf.sqrt(tf.reduce_sum(tf.square(vt), 1))
            #################
            self.vdotk = tf.reduce_sum(copied_vt*k, 2)
            self.normk = normk
            self.cp_vstar = copied_vt
            #################

            cos = - tf.reduce_sum(copied_vt*k, 2) / (normk * tf.reshape(normvt, [ped_count, 1]))
            wphi = tf.pow(((1 + cos) / 2), tf.reshape(b, [ped_count, 1]))
            E_i = tf.reduce_sum(wphi*wd*E_v, 1)

            ##################
            self.wphi = wphi
            self.cos = cos
            self.wd = wd
            self._E_i_op = E_i
            #################

            E = E_i + E_s + E_d
            E_total = tf.reduce_sum(E, 0)
            self._E_op = E_total

            optimizer = tf.train.AdamOptimizer(0.5)
            self._minimize_E_op = optimizer.minimize(E_total, var_list = [self._vstar])
            self._get_vstar_op = self._vstar

            #f = dE/dv
            f = tf.gradients(E_total, self._vstar)[0]

            vstar_dims = self._state_dims / 2 - 2
            f1 = []
            for index_vstar in range(vstar_dims):
                f1.append(tf.gradients(f[index_vstar], self._vstar)[0])
            dfdv = tf.pack(f1)

            f2 = []
            for index_vstar in range(vstar_dims):
                f2.append(tf.gradients(f[index_vstar], self._state)[0])
            dfds = tf.pack(f2)
            dvds = - tf.matmul(tf.matrix_inverse(dfdv),dfds)
            self._dv_dx_op = dvds

        self._optimal_vstar = tf.placeholder(tf.float64, [self._state_dims / 2 - 2])
        with tf.name_scope('move'):
            optimal_vstar = tf.reshape(self._optimal_vstar, [self._state_dims / 4 - 1, 2])
            org_agent_position = self._state[:2]
            org_agent_velocity = self._state[self._state_dims/2:self._state_dims/2+2]
            agent_velocity = org_agent_velocity + self._action * self._step_time
            agent_position = org_agent_position + (org_agent_velocity + agent_velocity) * self._step_time / 2.

            org_ped_position = self._state[2:self._state_dims/2]
            org_ped_velocity = self._state[self._state_dims/2+2:]
            ped_velocity = tf.reshape(optimal_vstar, [self._state_dims/2-2])
            alpha = tf.tile(a, [2])
            ped_position = org_ped_position + (alpha*org_ped_velocity + (1-alpha)*ped_velocity)*self._step_time

            self._next_state = tf.concat(0, [agent_position, ped_position, agent_velocity, ped_velocity])
            self._next_state_op = self._next_state

            # F = [dxt/dx|dxt/da] + [dxt/dv*]*[dv*/dx|dv*/da]
            F1 = []
            for state_index in range(self._state_dims):
                F1.append(tf.gradients(self._next_state[state_index], self._state)[0])
            dxtdx = tf.pack(F1)
            self._dxt_dx_op = dxtdx

            F2 = []
            for state_index in range(self._state_dims):
                F2.append(tf.gradients(self._next_state[state_index], self._action)[0])
            dxtda = tf.pack(F2)
            self._dxt_da_op = dxtda

            F3 = []
            for state_index in range(self._state_dims):
                F3.append(tf.gradients(self._next_state[state_index], self._optimal_vstar)[0])
            dxtdv = tf.pack(F3)
            self._dxt_dv_op = dxtdv
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
        ped_count = self._state_dims / 4 - 1
        ped_velocity = state[self._state_dims/2+2:].reshape((ped_count,2))
        initial_vstar = (ped_velocity + np.random.normal(0, 2.5, size = [ped_count,2])).reshape(-1)
        
        feed = {self._initial_vstar: initial_vstar}
        self._sess.run(self._initialize_vstar_op, feed)

        epsilon = 1.
        feed = {self._state: state, self._action:action}
        last_E = self._sess.run(self._E_op, feed)
        while epsilon > 1e-4:
            feed = {self._state: state, self._action:action}
            #_, E = self._sess.run([self._minimize_E_op, self._E_op], feed)
            self._sess.run(self._minimize_E_op, feed)
            E = self._sess.run(self._E_op, feed)
            epsilon = abs(E - last_E)
            last_E = E
        vstar = self._sess.run(self._get_vstar_op)
        feed = {self._state: state, self._action: action, self._optimal_vstar: vstar}
        next_state = self._sess.run(self._next_state_op, feed)
        # don't return cost, but return vstar
        return next_state, 0

    def linearize_step(self, state, action, vstar):
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
        feed = {self._initial_vstar: vstar}
        self._sess.run(self._initialize_vstar_op, feed)

        # F = [dxt/dx|dxt/da] + [dxt/dv*]*[dv*/dx|dv*/da]
        feed = {self._state: state, self._action:action, self._optimal_vstar: vstar}
        tic = time.time()
        dxtda = self._sess.run(self._dxt_da_op, feed)
        toc = time.time()
        print toc - tic

        tic = time.time()
        dxtdx = self._sess.run(self._dxt_dx_op, feed)
        toc = time.time()
        print toc - tic

        tic = time.time()
        dxtdv = self._sess.run(self._dxt_dv_op, feed)
        toc = time.time()
        print toc - tic

        tic = time.time()
        dvdx = self._sess.run(self._dv_dx_op, feed)
        toc = time.time()
        print toc - tic

        #dxtdx, dxtda, dxtdv, dvdx = self._sess.run([self._dxt_dx_op, self._dxt_da_op, self._dxt_dv_op, self._dv_dx_op], feed)
        dvda = np.zeros((dvdx.shape[0], 2))
        F = np.hstack((dxtdx, dxtda)) + np.dot(dxtdv, np.hstack((dvdx, dvda)))
        
        return F


class Initial_LTA_Controller:
    def __init__(self, base, model):
        self._base = base
        self._model = model
        self._state_dims = self._base.state_dims

    def generate_steps(self, initial_state):
        time_start = time.time()
        actions = []
        states = []
        states.append(initial_state)
        total_cost = 0
        done = False
        while not done:
            action = self._move(states[-1])
            next_state, cost = self._model.step(states[-1], action)
            total_cost += cost
            done = self._is_done(next_state)
            states.append(next_state)
            actions.append(action)
        #total_cost += self._model.final_cost(state)
        time_end = time.time()
        print "process time: ", time_end - time_start

        return actions, states
    def _is_done(self, state):
        agent_position = state[:2]
        agent_goal_position = self._base.agent.goal_position
        if np.linalg.norm(agent_position - agent_goal_position) < 10:
            return True

    def _move(self, state):
        """
        move all pedestrian except agent for one time step
        return state, action, cost, done
        """
        positions = state[:self._state_dims / 2]
        velocities = state[self._state_dims / 2:]
        agent_position = positions[:2]
        agent_velocity = velocities[:2]
        ped_position = np.array([]).reshape(0,2)
        ped_velocity = np.array([]).reshape(0,2)
        for ped_index in range(self._state_dims / 4 - 1):
            ped_position = np.vstack((ped_position, positions[ped_index*2 + 2 : ped_index*2 + 4]))
            ped_velocity = np.vstack((ped_velocity, velocities[ped_index*2 + 2 : ped_index*2 + 4]))

        rms_params = {}
        rms_params['gamma'] = 0.99
        rms_params['alpha'] = 0.5 #0.001
        rms_params['epsilon'] = 10**(-4)
        

        initial_velocity = agent_velocity.copy() + np.random.random((2)) * 2.5
        def energy_with_gradient(next_v):
            return self._energy_with_gradient(next_v, agent_velocity, agent_position, ped_velocity, ped_position)


        energy_list, minimize_energy_velocity = RMSprop(
                initial_velocity, energy_with_gradient, rms_params)
        action = (minimize_energy_velocity - agent_velocity) / self._base.step_time
        return action


    def _energy_with_gradient(self, next_v, agent_velocity, agent_position, ped_velocity, ped_position):
        """
        A function that calculates the energy and gradient of energy
        while given the velocity, the velocity is given
        in pixel unit, energy is in meters unit,
        and gradient of energy is in pixel unit
        Input:
            next_v: np.2darray; the velocity this pedestrian taken
        Return:
            energy: float; given this velocity, the energy for this pedestrian
            energy_gradient: np.2darray; the gradient of energy to the velocity
        """
        p2m = self._base.p2m
        v = p2m * next_v
        vt = p2m * agent_velocity
        u = p2m * self._base.agent.expected_speed
        l1 = self._base.agent.lambda1
        l2 = self._base.agent.lambda2
        sd = self._base.agent.sigma_d
        sw = self._base.agent.sigma_w
        b = self._base.agent.beta
        z = p2m * self._base.agent.goal_position
        p = p2m * agent_position

        # using back propogation in the following
        # E_s (speed) = lambda1 * (u - |v|)**2
        normv = np.linalg.norm(v) 
        E_s = l1 * (u - normv) ** 2

        gnormv = - 2 * l1 * (u - normv)
        gvs = gnormv * v / normv
        gvs2pixel = gvs * p2m

        # E_d (direction) = - (p dot v) / (|p| * |v|)
        pdotv = np.dot((z - p), v)
        normv = np.linalg.norm(v)
        normpnormv = np.linalg.norm((z-p)) * normv
        E_d = -l2 * pdotv / normpnormv

        gpdotv = -l2 / normpnormv
        gnormpnormv = l2 * pdotv / normpnormv ** 2
        gnormv = gnormpnormv * np.linalg.norm(z-p)
        gvd = gnormv * v / normv
        gvd += gpdotv * (z - p)
        gvd2pixel = gvd * p2m

        ped_count = ped_velocity.shape[0]

        gvi2pixel = np.array([0., 0.])
        E_i = 0.

        if ped_count != 0:
            # if there is more than one pedestrian (including agent), calculate social energy
            # agent's position subtracted by other pedestrian's position
            # agent's velocity subtracted by other pedestrian's velocity
            k = (agent_position - ped_position) * p2m
            q = (agent_velocity - ped_velocity) * p2m

            kdotq = np.sum(k * q, axis = 1) 
            normq = np.linalg.norm(q, axis = 1) 
            t = - kdotq / normq ** 2 #kdotq / |q|**2
            mask = t>0
            maskt = mask * t
            d = k + q * maskt[:, np.newaxis]
            normd = np.linalg.norm(d, axis = 1)
            E_v = np.exp( - normd**2 / (2 * sd**2))
            wd = np.exp(- np.linalg.norm(k, axis = 1)**2 / (2 * sw**2))
            cos = - np.dot(vt, np.swapaxes(k, 0, 1)) / (np.linalg.norm(vt) * np.linalg.norm(k, axis = 1))
            wphi = ((1 + cos) / 2)**b
            E_i = np.sum(wphi * wd * E_v)


            gE_v = wphi * wd
            gnormd = gE_v * E_v * (- normd / sd**2)
            gd = (gnormd / normd)[:, np.newaxis] * d
            gmaskt = np.sum(q * gd, axis = 1)
            gq = gd * maskt[:, np.newaxis]
            gt = gmaskt * mask
            gnormq = 2 *gt * kdotq / normq**3
            gq += (gnormq / normq)[:, np.newaxis] * q
            gkdotq = - gt / normq**2
            gq += gkdotq[:, np.newaxis] * k
            gvi = np.sum(gq, axis = 0)
            gvi2pixel = gvi * p2m

        # sum energy and energy gradient together
        energy = E_s + E_d + E_i 
        energy_gradient = gvs2pixel + gvd2pixel + gvi2pixel
        return (energy, energy_gradient)

def RMSprop(initial_value, energy_and_gradient_function, parameters):
    """
    Use RMSprop gradient descent to find the minimum of a function
    Input:
        initial_value: ?; the initial value to calculate the minimum by gradient descent
        energy_and_gradient_function: function; a function to calculate energy and its gradient given value 
                                      (same type as initial value)
        parameters: dictionary; a dictionary for all used parameters in RMSprop
            gamma, alpha (learning rate), epsilon: when to stop updating
    Return:
        energy_list: list of float; the process of gradient descent, shown by energy
        minimum_value: ?; the value that minimize the energy function
    """
    gamma = parameters['gamma']
    alpha = parameters['alpha']
    epsilon = parameters['epsilon']

    value = initial_value
    value_list = [initial_value]
    energy, energy_gradient = energy_and_gradient_function(value)
    energy_list = [energy]
    running_average = 0.0001

    done = False
    while not done:
        running_average = (1 - gamma) * np.square(energy_gradient) + running_average
        delta_value = alpha * energy_gradient / np.sqrt(running_average)
        value = value - delta_value
        value_list.append(value)
        updated_energy, energy_gradient = energy_and_gradient_function(value)
        delta_energy = abs(energy - updated_energy)
        if delta_energy < 10**-12 or abs(delta_energy / energy) < epsilon:
            done = True
        energy = updated_energy
        energy_list.append(energy)
    return (energy_list, value)

def test1():
    ped_params = {
        'lambda1': 2.33,
        'lambda2': 2.073,
        'sigma_d': 0.361,
        'sigma_w': 2.088,
        'beta': 1.462,
        'alpha': 0.73,
        'expected_speed': 50.,
        'goal_position': np.array([100., 300.000001])
        }
    agent_params = {
        'lambda1': 2.33,
        'lambda2': 2.073,
        'sigma_d': 0.361,
        'sigma_w': 2.088,
        'beta': 1.462,
        'expected_speed': 50.,
        'goal_position': np.array([700., 300.])
        }
    initial_state = np.array([100., 300.000001, 700., 300., 50., 0., -50., 0.])
    base = LTA_Base(8, agent_params, [ped_params])
    model = LTA_Model(base)
    controller = Initial_LTA_Controller(base, model)
    actions, states = controller.generate_steps(initial_state)
    display_states(base, states)

def test2():
    ped_params1 = {
        'lambda1': 2.33,
        'lambda2': 2.073,
        'sigma_d': 0.361,
        'sigma_w': 2.088,
        'beta': 1.462,
        'alpha': 0.73,
        'expected_speed': 50.,
        'goal_position': np.array([100., 320.000001])
        }
    ped_params2 = {
        'lambda1': 2.33,
        'lambda2': 2.073,
        'sigma_d': 0.361,
        'sigma_w': 2.088,
        'beta': 1.462,
        'alpha': 0.73,
        'expected_speed': 50.,
        'goal_position': np.array([100., 280.000001])
        }
    agent_params = {
        'lambda1': 2.33,
        'lambda2': 2.073,
        'sigma_d': 0.361,
        'sigma_w': 2.088,
        'beta': 1.462,
        'expected_speed': 50.,
        'goal_position': np.array([700., 300.])
        }
    initial_state = np.array([100., 300.000001, 700., 320., 700., 280., 50., 0., -50., 0., -50., 0.])
    base = LTA_Base(12, agent_params, [ped_params1, ped_params2])
    model = LTA_Model(base)
    controller = Initial_LTA_Controller(base, model)
    actions, states = controller.generate_steps(initial_state)
    display_states(base, states)

def test3():
    ped_count = 9
    random.seed(1.)
    ped_params = {
        'lambda1': 2.33,
        'lambda2': 2.073,
        'sigma_d': 0.361,
        'sigma_w': 2.088,
        'beta': 1.462,
        'alpha': 0.73,
        'expected_speed': 50.,
        'goal_position': np.array([100., 300.000001])
        }
    agent_params = {
        'lambda1': 2.33,
        'lambda2': 2.073,
        'sigma_d': 0.361,
        'sigma_w': 2.088,
        'beta': 1.462,
        'expected_speed': 50.,
        'goal_position': np.array([700., 300.])
        }
    ped_params_list = []
    ped_initial_position = []
    for i in range(ped_count):
        y = 300. + random.gauss(0., 100.)
        x_start = 700.
        x_end = 100.

        cp_ped_params = ped_params.copy()
        cp_ped_params['goal_position'] = np.array([x_end, y])

        ped_params_list.append(cp_ped_params)
        ped_initial_position.append(np.array([x_start, y]))

    initial_state = np.hstack([np.array([100., 300.000001]), np.hstack(ped_initial_position), np.array([50.,0.]), np.tile(np.array([-50.,0.]),ped_count)])
    base = LTA_Base((ped_count + 1)*4, agent_params, ped_params_list)
    model = LTA_Model(base)
    random.seed(time.time())
    controller = Initial_LTA_Controller(base, model)
    actions, states = controller.generate_steps(initial_state)
    display_states(base, states)
test3()
