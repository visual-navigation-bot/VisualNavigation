from __future__ import division
from __future__ import print_function

import numpy as np
import math
import os, sys
import random
import pygame
import pickle
import time
import cv2

from action_space.continuous_action_space import Continuous_Action_Space
from base_environment import Environment
from observation_space import Obs_Space_No_Ped_v0

PI = math.pi

class No_Ped_v0(Environment):
    def __init__(self, step_time, field_size, nav_path):
        # action range
        act_low = np.array([0.,-PI])
        act_high = np.array([2.,PI])

        # basic environment setting
        Environment.__init__(self, step_time, field_size)
        self._step_time = step_time
        self._W, self._H = field_size
        self._action_space = Continuous_Action_Space(act_low, act_high)
        self._observation_space = Obs_Space_No_Ped_v0(field_size) #(x,y)

        # load navigation data
        nav_path = os.path.abspath(os.path.expanduser(nav_path))
        with open(nav_path, 'rb') as f:
            nav_data = pickle.load(f)

        # navigation map used for reward from environment
        self._nmap = cv2.resize(nav_data['navigation_map'], (self._W,self._H))

        # load raw background image
        self._background_raw = cv2.resize(nav_data['background'], (self._W,self._H))

        # object color
        self._wall_color = (0,0,0)
        self._dest_color = (0,255,0)
        self._agent_color = (255,0,0)

        # start and destination
        self._start = self._random_sample_pos() #[y,x]
        self._destination = self._random_sample_pos()
        self._dest_hw = 10 # half window for destination

        self._background = self._background_raw.copy()
        self._add_dest2bg()

        # agent information, should be integers in range (H,W)
        self._x = 0
        self._y = 0

        # display
        self._circle_size = 5

    def step(self, action):
        # update agent position
        action[0] /= self._step_time
        hor_step = action[0] * math.cos(action[1])
        vert_step = action[0] * math.sin(action[1])
        self._x += hor_step
        self._y += vert_step

        x = int(self._x)
        y = int(self._y)

        obs = self._background.copy()

        # reward and terminal
        reach_dest = (x >= (self._destination[1]-self._dest_hw)) and (x <= self._destination[1]+self._dest_hw) and \
                     (y >= (self._destination[0]-self._dest_hw)) and (y <= self._destination[0]+self._dest_hw)
        try:
            out_of_bound = (x > self._W) or (x < 0) or (y > self._H) or (y < 0) or \
                           ((self._background[y-self._circle_size:y+self._circle_size, \
                                              x-self._circle_size:x+self._circle_size,:]==self._wall_color).all(2).any())
        except IndexError:
            out_of_bound = True
        if out_of_bound:
            # agent hit the wall
            r = -1000
            t = True
        elif reach_dest:
            # agent reach destination
            r = 1000
            t = True
        else:
            # observation, background + agent + destination
            obs[y,x,:] = self._agent_color
            # agent keep wandering
            r = -1 + self._nmap[y,x]*5
            t = False

        # display
        if self._display:
            self._display_clock_tick()
            # background
            self._screen.blit(self._background_display, [0,0])
            # agent
            pygame.draw.circle(self._screen, self._agent_color, (x, y), self._circle_size, 0)
            # flip
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    self._display = False
                    pygame.display.quit()
                    break

        return obs, r, t

    def display(self):
        # use pygame to display background + agent + destination
        self._display = True

        # display setting
        self._last_time = time.time()
        self._display_step_time = 1. / 20
        self._screen = pygame.display.set_mode((self._W, self._H))
        pygame.display.set_caption('No-Man-v0')

    def reset(self, random_start=True, random_dest=False):
        # start point is randomly initialized across episodes
        if random_start:
            self._start = self._random_sample_pos()

        # dest remains the same across different episodes
        if random_dest:
            self._dest = self._random_sample_pos()

        # add destination, background_raw --> background
        self._background = self._background_raw.copy()
        self._add_dest2bg()

        # add agent to background
        self._y, self._x = self._start[0], self._start[1]
        obs = self._background.copy()
        obs[self._y, self._x, :] = self._agent_color

        # display
        self._background_display = pygame.Surface((self._W,self._H))
        pygame.surfarray.array_to_surface(self._background_display, np.swapaxes(self._background, 0, 1))

        return obs

    def set_params(self, params):
        self._start = params['start']
        self._destination = params['destination']

    def _random_sample_pos(self):
        pos_map_x, pos_map_y = np.meshgrid(np.arange(self._W), np.arange(self._H))
        valid_pos = (self._background_raw != self._wall_color).all(2)
        pos_map_x = pos_map_x[valid_pos]
        pos_map_y = pos_map_y[valid_pos]
        pos = np.array([random.sample(pos_map_y,1)[0], random.sample(pos_map_x,1)[0]])

        return pos.astype(np.int32) #(y,x)

    def _add_dest2bg(self):
        # extend destination from a point to a region based on dest_hw
        y_min = max(0, int(self._destination[0]-self._dest_hw))
        y_max = min(self._H, int(self._destination[0]+self._dest_hw))
        x_min = max(0, int(self._destination[1]-self._dest_hw))
        x_max = min(self._W, int(self._destination[1]+self._dest_hw))
        if x_max!=x_min and y_max!=y_min:
            self._background[y_min:y_max, x_min:x_max, :] = self._dest_color

        # calibrate to avoid destination area cover obstacles
        non_wall = self._background_raw != self._wall_color
        self._background *= non_wall

    def _display_clock_tick(self):
        now_time = time.time()
        elap = self._display_step_time - (now_time-self._last_time)
        if elap>0:
            time.sleep(elap)
        self._last_time = time.time()

    def __str__(self):
        des = '?'
        
        return des
        
