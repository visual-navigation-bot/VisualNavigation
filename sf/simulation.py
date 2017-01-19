import pygame
import math
import random 
import numpy as np
FPS = 50 # tdiff is 50 fps

class Sim:
    #TODO add object interface
    def __init__(self, screen_name, screen_size):
        """
        Initialize a simulation environment
        """
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption(screen_name)
        self.screen.fill((255,255,255))
        self.clock = pygame.time.Clock()
        self.obstacles = []
        self.pedestrians = []

        self.TDIFF = 1. / FPS
        self.FPS = FPS

        # relationship between different ids
        self.PARTY_DIS = np.array([[]])
        self.PARTY_V = np.array([[]])
        self.PARTY_R = np.array([[]])
        self.FRIEND = np.array([[]])
        self.PID = {}

    def add_object(self, obj_type, param):
        """
        add an object to the simulation environment, either obstacle or pedestrian
        """
        if obj_type == 'ped':
            # add informations that is attribute of this class
            self.pedestrians.append(self.Ped(self, param))
            self.PID[param['pid']] = len(self.pedestrians) - 1

            # add on party information
            ped_num = len(self.pedestrians) - 1
            if ped_num == 0:
                self.FRIEND = np.array([[True]])
                self.PARTY_V = np.array([[0.]])
                self.PARTY_R = np.array([[0.]])
            else:
                party_v = self.PARTY_V
                V_others = param['V_others']
                v1 = np.insert(party_v, ped_num, V_others, axis = 1)
                v2 = np.append(V_others, 0)
                self.PARTY_V = np.insert(v1, ped_num, v2, axis = 0)

                party_r = self.PARTY_R
                R_others = param['R_others']
                r1 = np.insert(party_r, ped_num, R_others, axis = 1)
                r2 = np.append(R_others, 0)
                self.PARTY_R = np.insert(r1, ped_num, r2, axis = 0)


                party_f = self.FRIEND
                f_others = param['friend']
                f1 = np.insert(party_f, ped_num, f_others, axis = 1)
                f2 = np.append(f_others, True)
                self.FRIEND = np.insert(f1, ped_num, f2, axis = 0)
        if obj_type == 'obs':
            self.obstacles.append(self.Obstacle(self, param))
        return

    def remove_pedestrians(self, pid_list):
        """
        remove an pedestrians from the simulation environment
        pid_list is the list of pedestrian id (not index)
        """
        for pid in pid_list:
            idx = self.PID[pid]

            self.FRIEND = np.delete(self.FRIEND, idx, 0)
            self.FRIEND = np.delete(self.FRIEND, idx, 1)

            self.PARTY_V = np.delete(self.PARTY_V, idx, 0)
            self.PARTY_V = np.delete(self.PARTY_V, idx, 1)

            self.PARTY_R = np.delete(self.PARTY_R, idx, 0)
            self.PARTY_R = np.delete(self.PARTY_R, idx, 1)

            self.PARTY_DIS = np.delete(self.PARTY_DIS, idx, 0)
            self.PARTY_DIS = np.delete(self.PARTY_DIS, idx, 1)

            del self.pedestrians[idx]
            del self.PID[pid]

            for key, value in self.PID.items():
                if value > idx:
                    self.PID[key] = value - 1
        return



    def _get_dist_matrix(self):
        """
        calculate distance matrix between pedestrians
        """
        # calculate relative distance matrix to all pedestians
        ped_num = len(self.pedestrians)
        if ped_num != 0:
            s = self.pedestrians[0].s
            repeat = np.expand_dims(np.tile(s, (ped_num,1)), axis = 0)
            for pid in range(1, ped_num):
                s = self.pedestrians[pid].s
                r = np.expand_dims(np.tile(s, (ped_num,1)), axis = 0)
                repeat = np.concatenate((repeat, r))
            self.PARTY_DIS = np.swapaxes(repeat, 0, 1) - repeat

    def move(self):
        """
        update the location of pedestrians 
        """
        ped_num = len(self.pedestrians)
        if ped_num != 0:
            self._get_dist_matrix()
            
            # update pedestrians and display them
            for pid in range(ped_num):
                # relative information to other pedestrians
                other_dis = np.delete(self.PARTY_DIS[pid], pid, 0)
                other_V = np.delete(self.PARTY_V[pid], pid, 0)
                other_R = np.delete(self.PARTY_R[pid], pid, 0)

                self.pedestrians[pid].move(other_dis, other_V, other_R, self.TDIFF)

    def display(self):
        # display obstacles
        for obs in self.obstacles:
            obs.display(self.screen)

        for ped in self.pedestrians:
            ped.display(self.screen)


    def run(self):
        running = True
        while running:
            self.clock.tick(self.FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.move()
            self.display()
            pygame.display.flip()




    class Ped:
        def __init__(self, sim, param):
            """
            Initialize each pedestrians

            Input:
                sim: Sim; the simulator that contains this pedestrian
                param: dict; a dictionary for all parmaeters
                    init_s: np.1darray; initial location on map
                    exp_s: np.1darray; the target location of this pedestrian
                    v0: float; the expected velocity of that human
                    vmax: float; the maximum velocity of that human
                    init_v: np.1darray; initial velocity on the map
                    tau: float; expected time to accelerate to expected velocity
                    U: float; the obstacle potential energy level
                    R: float; the influence range of obstacles
            """
            self.sim = sim
            self.s = param['init_s']
            self.exp_s = param['exp_s']
            self.v0 = param['v0']
            self.vmax = param['vmax']
            self.v = param['init_v']
            self.tau = param['tau']
            self.V_obs = param['V_obs']
            self.R_obs = param['R_obs']
            self.sight_angle = param['sight_angle'] 
            self.sight_const = param['sight_const']
            self.dist_obs_func = param['dist_obs_func']
            self.pid = param['pid']


        def move(self, dist, V, R, tdiff):
            """
            Move the pedestrian in a time step

            Input:
                dist: np.2darray; relative location of the pedestrians
                V: np.1darray; potential energy to other pedestrians
                R: np.1darray; influence range to other pedestrians
                tdiff: float; time difference from last update
            """
            f = self.target_force() + self.repulsive_force(dist, V, R) + self.obstacle_force()
            self.v += f * tdiff
            norm_v = np.linalg.norm(self.v)
            if norm_v > self.vmax:
                self.v = self.v * self.vmax / norm_v
            self.s += self.v * tdiff
            
        def target_force(self):
            # force generated by the attraction from target
            e = (self.exp_s - self.s) / np.linalg.norm(self.exp_s - self.s)
            f = (self.v0 * e - self.v) / self.tau
            return f
        
        def obstacle_force(self):
            # force generated by the repulsion from target
            d = self.dist_obs_func(self.s)
            norm_d = np.linalg.norm(d)
            f = self.V_obs * d / (norm_d * self.R_obs) * np.exp(- norm_d / self.R_obs)
            return f

        def repulsive_force(self, other_dist, other_V, other_R):
            # force generated by the repulsion from other pedestrians
            d = np.linalg.norm(other_dist, axis = 1)
            const = - other_V * np.exp(-d / other_R) / (d * other_R)
            f = np.repeat(np.expand_dims(const, 1), 2, axis = 1) * other_dist

            # visualized domain tune
            e_v = self.v / np.linalg.norm(self.v)
            swp_f = np.swapaxes(f, 0, 1)
            swp_f = np.where(d * np.cos(self.sight_angle) > np.dot(other_dist, e_v), swp_f * self.sight_const, swp_f)
            f = np.swapaxes(swp_f, 0, 1)

            return np.sum(f, axis = 0)

        def attraction_force(self, friend_dis):
            # force generated by the attraction from other pedestrians
            pass


        def display(self, screen):
            # display the pedestrian on the screen
            pygame.draw.circle(screen, (0,0,0), (int(self.s[0]), int(self.s[1])), 1, 0)


    class Obstacle:
        def __init__(self, sim, param):
            """
            Define the location of the obstacle (now only for visualization)
            each obstacle is just a line

            Input:
                sim: Sim; the simulator that consists this obstacle
                param: dict; a dictionary for all parmaeters
                    'start': np.1darray; the start point on the map
                    'end': np.1darray; the end point on the map
            """
            self.start = param['start']
            self.end = param['end']

        def display(self, screen):
            # display the obstacle on the screen
            pygame.draw.line(screen, (0,0,0), (int(self.start[0]), int(self.start[1])), (int(self.end[0]), int(self.end[1])), 3)




