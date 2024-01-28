# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 13:57:14 2021

@author: x4nno
"""

import numpy as np
import matplotlib
import copy

SMASH_WALL = 4 # make sure to change to same as in environment


bridge_NS = np.array([[1, 1, 0, 0, 0, 1, 1],
                      [1, 1, 1, 0, 1, 1, 1],
                      [1, 1, 1, 0, 1, 1, 1],
                      [1, 1, 1, 0, 1, 1, 1],
                      [1, 1, 1, 0, 1, 1, 1],
                      [1, 1, 1, 0, 1, 1, 1],
                      [1, 1, 0, 0, 0, 1, 1]])

bridge_EW = np.array([[1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1],
                      [0, 1, 1, 1, 1, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 1, 1, 1, 1, 0],
                      [1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1]])

stairway_LR = np.array([[1, 1, 1, 1, 1, 0, 0],
                        [1, 1, 1, 1, 0, 0, 1],
                        [1, 1, 1, 0, 0, 1, 1],
                        [1, 1, 0, 0, 1, 1, 1],
                        [1, 0, 0, 1, 1, 1, 1],
                        [0, 0, 1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 1, 1, 1]])

stairway_RL = np.array([[0, 1, 1, 1, 1, 1, 1],
                        [0, 0, 1, 1, 1, 1, 1],
                        [1, 0, 0, 1, 1, 1, 1],
                        [1, 1, 0, 0, 1, 1, 1],
                        [1, 1, 1, 0, 0, 1, 1],
                        [1, 1, 1, 1, 0, 0, 1],
                        [1, 1, 1, 1, 1, 0, 0]])

connector_maze = np.array([[0, 1, 1, 0, 1, 1, 0],
                        [0, 0, 0, 0, 1, 0, 0],
                        [1, 0, 1, 0, 1, 0, 1],
                        [0, 0, 1, 0, 1, 0, 0],
                        [1, 0, 1, 0, 1, 0, 1],
                        [0, 0, 1, 0, 0, 0, 0],
                        [0, 1, 1, 0, 1, 1, 0]])

btrap_NS = np.array([[1, 1, 1, -2, 1, 1, 1],
                     [1, 1, 1, 0, 1, 1, 1],
                     [1, 1, 1, -2, 1, 1, 1],
                     [1, 1, 1, 0, 1, 1, 1],
                     [1, 1, 1, -2, 1, 1, 1],
                     [1, 1, 1, 0, 1, 1, 1],
                     [1, 1, 1, -2, 1, 1, 1]])

btrap_EW = np.array([[1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1],
                     [-2, 0, -2, 0, -2, 0, -2],
                     [1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1]])

river_EW = np.array([[1, 1, 1, 1, 1, 1, 1],
                     [1, -5, -5, -5, -5, -5, 1],
                     [1, -5, -5, -5, -5, -5, 1],
                     [0, -5, -5, -5, -5, -5, 0],
                     [1, -5, -5, -5, -5, -5, 1],
                     [1, -5, -5, -5, -5, -5, 1],
                     [1, -1, -1, -1, -1, -1, 1]])

river_NS = np.array([[1, 1, 1, 0, 1, 1, 1],
                     [-1, -3, -3, -3, -3, -3, 1],
                     [-1, -3, -3, -3, -3, -3, 1],
                     [-1, -3, -3, -3, -3, -3, 1],
                     [-1, -3, -3, -3, -3, -3, 1],
                     [-1, -3, -3, -3, -3, -3, 1],
                     [1, 1, 1, 0, 1, 1, 1]])

# Change this centre 0 to WALL_BREAK or something value at the top for this function

breakdown_NS = np.array([[1, 1, 1, 0, 1, 1, 1],
                         [1, 1, 1, 0, 1, 1, 1],
                         [1, 0, 0, 0, 0, 0, 1],
                         [1, 1, 1, 0, 1, 1, 1],
                         [1, 0, 0, 0, 0, 0, 1],
                         [1, 1, 1, 0, 1, 1, 1],
                         [1, 1, 1, 0, 1, 1, 1]])

breakdown_EW = np.array([[1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 0, 1, 0, 1, 1],
                         [1, 1, 0, 1, 0, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0],
                         [1, 1, 0, 1, 0, 1, 1],
                         [1, 1, 0, 1, 0, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1]])

empty = np.array([[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]])

walls = np.array([[1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1]])

four_rooms = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1],
                      [1,0,0,0,0,0,0,1,0,0,0,0,1],
                      [1,0,0,0,0,0,0,1,0,0,0,0,1],
                      [1,0,0,0,0,0,0,0,0,0,0,0,1],
                      [1,0,0,0,0,0,0,1,0,0,0,0,1],
                      [1,0,0,0,0,0,0,1,0,0,0,0,1],
                      [1,1,1,0,1,1,1,1,1,1,0,1,1],
                      [1,0,0,0,0,0,1,0,0,0,0,0,1],
                      [1,0,0,0,0,0,1,0,0,0,0,0,1],
                      [1,0,0,0,0,0,0,0,0,0,0,0,1],
                      [1,0,0,0,0,0,1,0,0,0,0,0,1],
                      [1,0,0,0,0,0,1,0,0,0,0,0,1],
                      [1,1,1,1,1,1,1,1,1,1,1,1,1]])

Josh_grid = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                [1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1],
                [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                [1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1],
                [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
                )




def create_Tori_bbs():
    # Create a 7x7 array filled with zeros
    arr = np.zeros((7, 7), dtype=int)
    
    # Choose a random row and column to place the 1
    # row = np.random.randint(0, 7)
    # col = np.random.randint(0, 7)
    
    bbs_list = []
    for r in range(1,7):
        for c in range(1,7):
            new_arr = copy.copy(arr)
            new_arr[:,c] = 1
            new_arr[r,c] = 0
            bbs_list.append(new_arr)
    # Set the value at the chosen position to 1
    
    return bbs_list


def create_four_rooms_bbs():
    bbs_list = [four_rooms] # because of the "reward blocks" later.
    return bbs_list

def create_Josh_grid():
    bbs_list = [Josh_grid] # because of the "reward blocks" later
    return bbs_list

def get_building_blocks(time=False, style="grid"):
    
    if style == "Tori":
        return create_Tori_bbs()   
    
    elif style == "Four_rooms":
        return create_four_rooms_bbs()
    
    elif style == "Josh_grid":
        return create_Josh_grid()
    
    elif style == "grid":
        if time:
            return [bridge_NS, bridge_EW, stairway_LR, stairway_RL, btrap_NS, btrap_EW,
                    river_EW, river_NS, breakdown_NS, breakdown_EW, connector_maze, empty]  # add empty in to make easier
        else:
            return [bridge_NS, bridge_EW, stairway_LR, stairway_RL, breakdown_NS, breakdown_EW, connector_maze, empty]


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    bb = get_building_blocks(style="grid")
    
    colors = ["white", "black", "blue", "yellow", "brown"]
    levels = [0, 1, 2, 3, 4, 5]
    
    cm, norm = matplotlib.colors.from_levels_and_colors(levels, colors)
    
    print(cm)
    

    for b in bb:
        plt.imshow(b, cmap=cm, norm=norm, interpolation="none")
        plt.show()

    # key here:

    # -1 = lava
    # 1 = wall
    # -2 = btrap_spikes #these invert between negative reward and empty space every step
    # -3 = fast flowing water to the left #effect of blowing 1 block every two moves in the river
    # -4 = breakdown #three hits into the wall will cause the wall to break through
    # -5 = fast flowing water to the south # effect of blowing 1 block every two moves in the river
    # 0 = empty
    # 2 = agent location
    # 3 = goal location
