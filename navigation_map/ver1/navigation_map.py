from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import pickle
import os
import math

# follows BGR (opencv style)
BLACK = (0,0,0)
WHITE = (255,255,255)
BLUE = (255,0,0)
GREEN = (0,255,0)
RED = (0,0,255)
ORANGE = (0,109,255)
YELLOW = (255,255,0)
PURPLE = (128,0,128)
CYAN = (0,255,255)
PINK = (105,0,255)
GRAY = (120,120,120)

def search_dict_key(dict_in, val_in):
    for k, v in dict_in.items():
        if v==val_in:
            return k

class NavigationMap(object):
    def __init__(self, nmap_conf):
        '''
        FUNC: constructor of NavigationMap
        Arguments:
            nmap_conf: configuration of navigation map, must have keys
                       'width': width of the navigation map, 
                       'height': height of the navigation map, and
                       'patch_size': size of patches which the navigation map wil be divied into
        '''
        # configuration of navigation map
        self._patch_size = nmap_conf['patch_size']
        self._width = nmap_conf['width']
        self._height = nmap_conf['height']

        # height and width of the grid consisting of patches
        self._patch_H = int(math.ceil(self._height/self._patch_size))
        self._patch_W = int(math.ceil(self._width/self._patch_size))
    
        # direction map with each element as an integer representing a direction
        self._dir_map = np.zeros((self._patch_H, self._patch_W), dtype=np.uint8)
        # dictionary for direction map
        self._dmap_dict = {
            'undefined': 0,
            'not_allowed': 1,
            'north': 2,
            'east': 3,
            'south': 4,
            'west': 5,
            'north_east': 6,
            'south_east': 7,
            'south_west': 8,
            'north_west': 9
        }

        # visualization of allow_map and dir_map
        self._vis_dmap = 255 * np.ones((self._height, self._width, 3), dtype=np.uint8)
        self._drawn_dmap = self._vis_dmap.copy()
        self._dmap_color = { # specify direction colors
            'undefined': WHITE,
            'not_allowed': BLACK,
            'north': BLUE,
            'east': GREEN,
            'south': PINK,
            'west': ORANGE,
            'north_east': YELLOW,
            'south_east': PURPLE,
            'south_west': GRAY,
            'north_west': CYAN
        }
        self._cur_label = 'undefined'
        self._rect_color = RED # specify rectangle color
        self._rect = (0,0,1,1)
        self._rect_hold = False
        self._rect_over = False
        self._left_dclk = False
        self._ix = 0
        self._iy = 0

    def visualize(self):
        cv2.namedWindow('navigation map visualization', cv2.WINDOW_NORMAL)
        cv2.imshow('navigation map visualization', self._vis_dmap)
        cv2.waitKey(0)

    def edit(self):
        '''
        FUNC: edit navigation map using mouse click and command line input
        '''
        print('Start editing mode:')
        print('    press \'h\' to get help message')
        print('    press \'l\' to get color-direction correspondence')
        print('    press \'p\' to print out current label')
        print('    press \'s\' to set label')
        print('    press \'q\' or \'esc\' to quit editing')
        print('')
        cv2.namedWindow('Edit Navigation Map', cv2.WINDOW_NORMAL) #WINDOW_AUTOSIZE
        cv2.setMouseCallback('Edit Navigation Map', self._onmouse)
        while(True):
            cv2.imshow('Edit Navigation Map', self._drawn_dmap)

            if self._rect_over:
                # A rectangle was drawn. Do something
                # obtain patches covered by rect
                px1, py1, px2, py2 = (np.array(self._rect)/self._patch_size).astype(np.uint8)
                px2 += 1
                py2 += 1
                # update dir_map
                self._dir_map[py1:py2,px1:px2] = self._dmap_dict[self._cur_label]
                # update vis_dmap
                self._vis_dmap[py1*self._patch_size:py2*self._patch_size, px1*self._patch_size:px2*self._patch_size] \
                                 = self._dmap_color[self._cur_label]
                # update display image and flag
                self._drawn_dmap = self._vis_dmap.copy()
                self._rect_over = False

            if self._left_dclk:
                # A left double click happened. Do something
                # obtain patch x, y
                px = int(self._ix/self._patch_size)
                py = int(self._iy/self._patch_size)
                # update dir_map
                self._dir_map[py,px] = self._dmap_dict[self._cur_label]
                # update vis_dmap
                self._vis_dmap[py*self._patch_size:(py+1)*self._patch_size, px*self._patch_size:(px+1)*self._patch_size] \
                                  = self._dmap_color[self._cur_label]
                # update display image and flag
                self._drawn_dmap = self._vis_dmap.copy()
                self._left_dclk = False

            key = cv2.waitKey(10)
            if key & 0xFF == ord('h'):
                # print help message
                print('')
                print('press \'h\' to get help message')
                print('press \'l\' to get color-direction correspondence')
                print('press \'p\' to print out current label')
                print('press \'s\' to set label')
                print('press \'q\' or \'esc\' to quit editing')
                print('')
            elif key & 0xFF == ord('l'):
                # print color-direction correspondence
                print('')
                print('undefined = WHITE')
                print('not_allowed = BLACK')
                print('north = BLUE')
                print('east =  GREEN')
                print('south = PINK')
                print('west = ORANGE')
                print('north_east = YELLOW')
                print('south_east = PURPLE')
                print('south_west = GRAY')
                print('north_west = CYAN')
                print('')
            elif key & 0xFF == ord('p'):
                print('Current label = {}'.format(self._cur_label))
            elif key & 0xFF == ord('s'):
                # set current label (undefined + not_allowed + 8_directions)
                label_in = raw_input('>>> Set label to: ')
                try:
                    self._dmap_dict[label_in]
                    self._cur_label = label_in
                except KeyError:
                    print('No such label in direction map. Please enter \'l\' to check valid labels.')
            elif (key & 0xFF == ord('q')) or (key & 0xFF == 27):
                # quit video
                print('End editing')
                break

        cv2.destroyWindow('Edit Navigation Map')

    def save(self, f_name):
        '''
        FUNC: save navigation map and its configuration to .pkl file
        Argument:
            f_name: filename to be saved as
        '''
        data = {
            'patch_size': self._patch_size,
            'width': self._width,
            'height': self._height,
            'dir_map': self._dir_map,
            'dmap_dict': self._dmap_dict
        } 
        f_name = os.path.abspath(os.path.expanduser(f_name))
        with open(f_name, 'wb') as f:
            pickle.dump(data,f)
        
    def restore(self, f_name):
        '''
        FUNC: restore navigation map from .pkl file
              NOTE that it may cover some settings from __init__
        Argument:
            f_name: filename specified from which navigation map is restored
        '''
        f_name = os.path.abspath(os.path.expanduser(f_name))
        with open(f_name, 'rb') as f:
            data = pickle.load(f)

        self._patch_size = data['patch_size']
        self._width = data['width']
        self._height = data['height']
        self._dir_map = data['dir_map']
        self._dmap_dict = data['dmap_dict']

        self._patch_H = int(self._height/self._patch_size) + 1
        self._patch_W = int(self._width/self._patch_size) + 1
        self._vis_dmap = self._dmap2vis(data['dir_map'])
        self._drawn_dmap = self._vis_dmap.copy()
        
    def _onmouse(self, event, x, y, flags, param):
        '''
        FUNC: mouse callback function (opencv style)
        '''
        # draw rectangular
        if event == cv2.EVENT_LBUTTONDOWN:
            self._rect_hold = True
            self._ix, self._iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._rect_hold:
                self._drawn_dmap = self._vis_dmap.copy()
                cv2.rectangle(self._drawn_dmap, (self._ix,self._iy), (x,y), self._rect_color, 2)
        elif event == cv2.EVENT_LBUTTONUP:
            self._rect_hold = False
            self._rect_over = True
            cv2.rectangle(self._drawn_dmap, (self._ix,self._iy), (x,y), self._rect_color, 2)
            self._rect = (min(self._ix,x), min(self._iy,y), max(self._ix,x), max(self._iy,y))
            
            bRect0 = max(0, self._rect[0])
            bRect1 = max(0, self._rect[1])
            bRect2 = min(self._width, self._rect[2])
            bRect3 = min(self._height, self._rect[3])
            self._rect = tuple((bRect0,bRect1,bRect2,bRect3))
        # double click to assign value
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            self._ix, self._iy = x, y
            self._left_dclk = True

    def _dmap2vis(self, dmap):
        '''
        FUNC: consturct vis_dmap from dir_map
        '''
        vis_dmap = np.zeros((self._height,self._width,3), dtype=np.uint8)
        psz = self._patch_size
        for i in range(self._patch_H):
            for j in range(self._patch_W):
                label = search_dict_key(self._dmap_dict, dmap[i,j])
                vis_dmap[i*psz:(i+1)*psz, j*psz:(j+1)*psz] = self._dmap_color[label]

        return vis_dmap
        
    @property
    def width(self):
        return self._width
    @property
    def height(self):
        return self._height
    @property
    def patch_size(self):
        return self._patch_size
    @property
    def dir_map(self):
        return self._dir_map

