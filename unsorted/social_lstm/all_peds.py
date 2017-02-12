from __future__ import division

import tensorflow as tf
import numpy as np
import os, sys
import pickle
import argparse
import random
import math
import pdb#DEBUG
import time

import utils
from social_model import SocialModel
from grid import getSequenceGridMask

# args: social_conf_path, ckpt_dir, new_peds_prob, bg_shape, init_num_step
class AllPeds(object):
    def __init__(self, args, verbose=True):
        self.verbose = verbose
    
        # define social LSTM model
        print('Building social LSTM model')
        with open(utils.check_path(args.social_conf_path), 'rb') as f:
            self._social_conf = pickle.load(f)
        self._model = SocialModel(self._social_conf, True)
        
        # define session
        self._sess = tf.InteractiveSession()
        
        # restore model parameters
        restorer = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(os.path.abspath(os.path.expanduser(args.ckpt_dir)))
        print('loading model: {}'.format(ckpt.model_checkpoint_path))
        restorer.restore(self._sess, ckpt.model_checkpoint_path)
        
        # probability of a new pedestrian pops up if self._cur_num_peds doesn't reach self._max_num_peds
        self._new_peds_prob = args.new_peds_prob
        # maximum number of pedestrian in a frame
        self._max_num_peds = self._social_conf.maxNumPeds
        # number of pedestrian in the current frame
        self._cur_num_peds = 0
        # a list to indicate which pedestrians among all max_num_peds pedestrian exist
        self._peds_exist = [False]*self.max_num_peds
        # internal data for social LSTM model
        self._data = np.zeros((1,self._max_num_peds,3)) # shape=(1,MNP,3)
        self._grid_data = np.zeros((1,self._max_num_peds,self._max_num_peds,
                                    self._social_conf.grid_size**2)) # shape=(1,MNP,MNP,grid_size**2)
        self._init_data = np.zeros((args.init_num_step,self._max_num_peds,3)) # shape=(init_num_step,MNP,3)
        self._init_grid_data = np.zeros((args.init_num_step,self._max_num_peds,self._max_num_peds,
                                         self._social_conf.grid_size**2))  # shape=(init_num_step,MNP,MNP,grid_size**2)
        # shape of background, a 2-element list [width, height]
        self._bg_shape = args.bg_shape
        # number of step for initialization of a pedestrian
        self._init_num_step = args.init_num_step

        # for interpolation
        self._n_interp = args.n_interp
        self._interp_count = 0
        self._prev_data = np.zeros(self._data.shape)

        self._output_data = np.zeros(self._data.shape)

    def step(self):
        if self._interp_count%self._n_interp==0:
            self._prev_data = self._data
            self._step()
            self._output_data = self._prev_data
        else:
            xy_data_prev = self._prev_data[0,:,1:3]
            #xy_data_prev[self._data[0,:,0]==0,:] = 0
            xy_data = self._data[0,:,1:3]

            self._output_data[0,:,0] = self._data[0,:,0]
            r = (self._interp_count+1.)/self._n_interp
            self._output_data[0,:,1:3] = (1-r)*xy_data_prev + r*xy_data

        self._interp_count = (self._interp_count+1)%self._n_interp

    def _step(self):
        ### a certain chance of a new pedstrain poping up
        # only possible if current number of pedestrians doesn't exceed the maximum number
        if self._cur_num_peds<self._max_num_peds:
            new_ped_pops_up = self._pops_up_fn(self._cur_num_peds, self._max_num_peds)
            if new_ped_pops_up:
                # create initial data for the new pedestrain
                new_data = self._init_ped_data()
                # add data of the new pedestrian self._data and self._grid_data
                for i in range(self._max_num_peds):
                    if self._data[0,i,0]==0: # an unoccupied element
                        newly_exist_pedID = i + 1
                        self._init_data[:,i,0] = newly_exist_pedID
                        self._init_data[:,i,1:3] = new_data
                        break
                assert(newly_exist_pedID==i+1)#DEBUG
                self._init_grid_data = getSequenceGridMask(self._init_data, self._bg_shape,
                                                           self._social_conf.neighborhood_size,
                                                           self._social_conf.grid_size)
                # reinitialize LSTM model
                self._model.sample_init(self._sess, self._init_data, self._init_grid_data)
                self._data[0,i,:] = self._init_data[-1,i,:] #np.reshape(self._init_data[-1,:,:], self._data.shape)
                self._grid_data = np.reshape(self._init_grid_data[-1,:,:,:], self._grid_data.shape)
                # update current number of pedestrians and pedestrian existence list
                self._cur_num_peds += 1
                self._peds_exist[i] = True

                if self.verbose:
                    print('A new pedestrian with ID {} pops up at ({},{})'\
                          .format(newly_exist_pedID,
                                  int(new_data[0,0]*self._bg_shape[0]), 
                                  int(new_data[0,1]*self._bg_shape[1])))
        ### predict next step of all existing pedestrians
        self._data, self._grid_data = self._model.sample_one_step(self._sess, self._data,
                                                                  self._grid_data, self._bg_shape)

        ### remove pedestrians out-of-bound (not in the background area)
        for i in range(self._max_num_peds):
            pedID = self._data[0,i,0]
            # if pedID==0 --> nonexisting pedestrian
            if pedID!=0:
                x = self._data[0,i,1]
                y = self._data[0,i,2]
                if (x<-0.1) or (x>1.1) or (y<-0.1) or (y>1.1):
                    # remove data of current pedstrian from self._data and self._grid_data
                    self._init_data[:,i,:] = 0
                    self._init_grid_data = getSequenceGridMask(self._init_data, self._bg_shape,
                                                               self._social_conf.neighborhood_size,
                                                               self._social_conf.grid_size)
                    # reinitialize social LSTM model
                    self._model.sample_init(self._sess, self._init_data, self._init_grid_data)
                    self._data[0,i,:] = self._init_data[-1,i,:] #np.reshape(self._init_data[-1,:,:], self._data.shape)
                    self._grid_data = np.reshape(self._init_grid_data[-1,:,:,:], self._grid_data.shape)
                    # update current number of pedestrian and pedestrian existence list
                    self._cur_num_peds -= 1
                    self._peds_exist[i] = False
                    
                    if self.verbose:
                        print('A pedestrian with ID {} is out-of-bound at ({},{}) and is removed.'\
                              .format(int(pedID),
                                      int(x*self._bg_shape[0]), 
                                      int(y*self._bg_shape[1])))

    def _pops_up_fn(self, cur_n, max_n):
        #prob = self._new_peds_prob
        if self._cur_num_peds<=15:
            prob = 0.5
        else:
            prob = -0.1
        rv = random.uniform(0,1) # a random variable ~ U(0,1)
        coin = (rv<=prob)
        return coin

    def _init_ped_data(self):
        random.seed(time.time())
        data = np.zeros((self._init_num_step,2))
        # randomly pick a side among 4 sides of the background
        which_side = random.randint(1,4)
        # randomly select start speed
        #mu = 0.01
        #sigma = 0.005
        #speed = np.random.normal(mu, sigma, 1)[0]
        speed = random.uniform(0.0001,0.0005) / 10000
        # take steps (not considering direction yet)
        scalar_steps = speed * np.arange(self._init_num_step)
        # walking direction
        angle = random.uniform(0,math.pi/2)
        # random start point
        start = random.uniform(0.2,0.8)
        # calibrate angle according to different sides from wich the ped pops up
        if which_side==1: # up
            angle += 1.25 * math.pi
            data[:,0] = scalar_steps * math.cos(angle) + start # x
            data[:,1] = scalar_steps * -math.sin(angle) # y 
        elif which_side==2: # left
            angle += 1.75
            if angle>=2.: 
                angle -= 2.
            data[:,0] = scalar_steps * math.cos(angle) # x
            data[:,1] = scalar_steps * math.sin(angle) + start # y
        elif which_side==3: # down
            angle += 0.25
            data[:,0] = scalar_steps * math.cos(angle) + start # x
            data[:,1] = 1. - scalar_steps * math.sin(angle) # y
        else: # which_side==4, right
            angle += 0.75
            data[:,0] = 1. + scalar_steps * math.cos(angle) # x
            data[:,1] = scalar_steps * math.sin(angle) + start # y

        return data
   
    def check_collision(self):
        tol = 1./np.amax(self._bg_shape)
        n_collision = 0
        for i in range(self._max_num_peds):
            pedID_ref = self._data[0,i,0]
            if pedID_ref!=0:
                for j in range(i+1,self._max_num_peds):
                    pedID_other = self._data[0,j,0]
                    if pedID_other!=0:
                        refXY = self._data[0,i,1:3]
                        otherXY = self._data[0,j,1:3]
                        dist = np.sum(np.square(refXY-otherXY))**0.5
                        if dist<tol:
                            n_collision += 1
                            if self.verbose:
                                print('Pedestrian {} collides with pedestrian {}'\
                                      .format(int(pedID_ref), int(pedID_other)))
        if self._cur_num_peds>1:
            c_rate = (2*n_collision/((self._cur_num_peds**2)*(self._cur_num_peds-1)))**0.5
        else:
            c_rate = 0.

        return n_collision, c_rate

    def close_session(self):
        self._sess.close()
        print('TF session in AllPeds class is closed.')

    @property
    def bg_shape(self):
        return self._bg_shape
    @property
    def max_num_peds(self):
        return self._max_num_peds
    @property
    def cur_num_peds(self):
        return self._cur_num_peds
    @property
    def existing_peds(self):
        existing_peds = self._output_data[0,self._output_data[0,:,0]!=0,:]
        existing_peds[:,0] = (existing_peds[:,0])
        existing_peds[:,1] = (existing_peds[:,1]*self._bg_shape[0]).astype(np.int32)
        existing_peds[:,2] = (existing_peds[:,2]*self._bg_shape[1]).astype(np.int32)
        return existing_peds

