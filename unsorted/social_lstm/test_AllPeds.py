from __future__ import division

import numpy as np
import argparse
import time
import pdb
import pickle
import pygame

from all_peds import AllPeds

def test1():
    total_step = 100

    parser = argparse.ArgumentParser()
    args = parser.parse_args() 
    args.social_conf_path = './save/social_config.pkl'
    args.ckpt_dir = './save'
    args.new_peds_prob = 1.0
    args.bg_shape = [600,400]
    args.init_num_step = 5
    args.n_interp = 4

    tic = time.clock()
    AP = AllPeds(args, verbose=True)
    toc = time.clock()
    print('\n')
    print('********* initial condition ************')
    print('Background shape (width,height): ({}, {})'.format(AP.bg_shape[0],AP.bg_shape[1]))
    print('Maximum number of pedestrians: {}'.format(AP.max_num_peds))
    print('Number of pedestrians at current step: {}'.format(AP.cur_num_peds))
    print('Model construction time: {:.3}'.format(toc-tic))

    total_collision = 0
    total_c_rate = 0.
    for i in range(total_step):
        print('\n')
        print('************** step {} **************'.format(i))
        tic = time.clock()
        AP.step()
        toc = time.clock()
        print('cur_num_peds: {}'.format(AP.cur_num_peds))
        print('existing pedestrian data(ID,x,y): \n{}'.format(AP.existing_peds))
        print('Elapsed time: {:.3}'.format(toc-tic))
        n_collision, c_rate = AP.check_collision()
        total_collision += n_collision
        total_c_rate += c_rate
    ratio = total_collision / total_step
    mean_c_rate = total_c_rate / total_step
    print('(total number of collisions)/(total steps) = ({}/{}) = {}'\
          .format(total_collision, total_step, ratio))
    print('mean collision rate = {:.5}'.format(mean_c_rate))

    AP.close_session()    

def compute_c_rate(total_step, print_step):
    parser = argparse.ArgumentParser()
    args = parser.parse_args() 
    args.social_conf_path = './save/social_config.pkl'
    args.ckpt_dir = './save'
    args.new_peds_prob = 1.0
    args.bg_shape = [600,400]
    args.init_num_step = 5
    args.n_interp = 4

    tic = time.clock()
    AP = AllPeds(args, False)
    toc = time.clock()
    print('\n')
    print('********* initial condition ************')
    print('Background shape (width,height): ({}, {})'.format(AP.bg_shape[0],AP.bg_shape[1]))
    print('Maximum number of pedestrians: {}'.format(AP.max_num_peds))
    print('Number of pedestrians at current step: {}'.format(AP.cur_num_peds))
    print('Model construction time: {:.3}'.format(toc-tic))
    print('\n')

    total_collision = 0
    total_c_rate = 0.
    c_rate_list = []
    for i in range(total_step):
        AP.step()
        n_collision, c_rate = AP.check_collision()
        total_collision += n_collision
        total_c_rate += c_rate
        if i%print_step==0:
            cur_c_rate = total_c_rate/(i+1)
            print('current c_rate = {:.5}'.format(cur_c_rate))
            c_rate_list.append(cur_c_rate)

    ratio = total_collision / total_step
    mean_c_rate = total_c_rate / total_step
    print('(total number of collisions)/(total steps) = ({}/{}) = {}'\
          .format(total_collision, total_step, ratio))
    print('mean collision rate = {:.5}'.format(mean_c_rate))
    c_rate_list.append(mean_c_rate)

    AP.close_session() 

    return c_rate_list

def visualize():
    color = (0,0,255)
    radius = 4
    thick = 3

    parser = argparse.ArgumentParser()
    args = parser.parse_args() 
    args.social_conf_path = './save/social_config.pkl'
    args.ckpt_dir = './save'
    args.new_peds_prob = 1.0
    args.bg_shape = [600,400]
    args.init_num_step = 5
    args.n_interp = 4

    tic = time.clock()
    AP = AllPeds(args, True)
    toc = time.clock()
    print('\n')
    print('********* initial condition ************')
    print('Background shape (width,height): ({}, {})'.format(AP.bg_shape[0],AP.bg_shape[1]))
    print('Maximum number of pedestrians: {}'.format(AP.max_num_peds))
    print('Number of pedestrians at current step: {}'.format(AP.cur_num_peds))
    print('Model construction time: {:.3}'.format(toc-tic))
    print('\n')

    screen = pygame.display.set_mode(tuple(args.bg_shape))
    pygame.display.set_caption('Social LSTM visualization')   
    clock = pygame.time.Clock()
    running = True

    while(running):
        clock.tick(20)
        # reset screen
        screen.fill((255,255,255))
        # take a step
        AP.step()
        peds = AP.existing_peds
        peds = peds[:,1:3].astype(np.int32)
        # display
        for ped in peds:
            pygame.draw.circle(screen, color, tuple(ped), radius, thick)
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

    AP.close_session()

#test1()

#c_rate_list = compute_c_rate(10000, 100) # (total_step, print_step)
#with open('results/c_rate_list.pkl', 'wb') as f:
#    pickle.dump(c_rate_list, f)

visualize()

