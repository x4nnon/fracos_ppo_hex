#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:20:07 2023

@author: x4nno
"""

import pygame
import numpy as np
import Environment_obstacles
from gym import spaces as spaces_old
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import copy
from collections import OrderedDict
import torch
from matplotlib import pyplot as plt

class MetaGridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=7, agent_location_value=2,
                goal_location_value=3, domain_size=[21,21], 
                stochastic=False, stochastic_strength=None,
                max_episode_steps=500, style="grid", seed=0):
        
        self.stochastic = stochastic
        self.stochastic_strength = stochastic_strength
        self.agent_location_value = agent_location_value
        self.goal_location_value = goal_location_value
        self.style = style
        
        self.rand_seed = seed
        
        self.env_master = Environment_obstacles.Environment(self.agent_location_value,
                                                     self.goal_location_value,
                                                     domain_size=domain_size, 
                                                     max_episode_steps=max_episode_steps,
                                                     style=self.style, seed=self.rand_seed)
        
        self.done = self.env_master.done
        
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Box(0, 7, shape=(51,), dtype=float)
        
        # spaces.Dict(
        #     {
        #         "view": spaces.Box(0, 4, shape=(7,7), dtype=int),
        #         "dir_mag": spaces.Box(0, size, shape=(2,), dtype=float),
        #     }
        # )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.discrete.Discrete(4)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        
        # make sure these are lists
        self.action_translation = {0:('up',), 1:('down',), 2:('left',), 3:('right',)}
        
    def seed(self, random_seed):
        self.rand_seed = random_seed
        self.env_master.seed(random_seed)
        
    def _get_obs(self):
        view, dir_mag = self.env_master.get_observation_space()
        view = view.flatten()
        return np.concatenate((view, dir_mag))
        # return {"view": view, "dir_mag": dir_mag}
    
    def get_obs(self):
        view, dir_mag = self.env_master.get_observation_space()
        view = view.flatten()
        return np.concatenate((view, dir_mag))
        # return {"view": view, "dir_mag": dir_mag}
    
    def _get_info(self):
        return {"info": "Info not implemented"}
    
    def get_info(self):
        return {"info": "Info not implemented"}
    
    def reset(self, goal_choice=[], start_same=False, total_random=False, seed=None, options=None):
        if seed:
            self.seed(seed)
        
        self.env_master.reset(goal_choice=goal_choice,
                       start_same=start_same, 
                       total_random=total_random)
        
        self.done = self.env_master.done
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def render(self):
        plt.imshow(self.env_master.domain)
        plt.show()
        plt.close()
    
    def step(self, action):
        
        if not isinstance(action, int):
            action = int(action) # take from tensor.
            
        action = self.action_translation[action]
        domain, reward, terminated, info = self.env_master.step(action)
        info = {"not implemented":info}
        
        
        if terminated == True:
            obs_space = copy.deepcopy(self.env_master.domain)
            obs_space = np.pad(obs_space, ((3,3),(3,3)), "constant", constant_values=((1,1),(1,1)))
            obs_centre = Environment_obstacles.find_agent_location(obs_space)
            obs_space_1 = obs_space[obs_centre[0]-3:obs_centre[0]+4, obs_centre[1]-3:obs_centre[1]+4]
            obs_space_1[3][3] = 0
            
            if np.where(obs_space_1==3)[0].size != 0:
                goal_x = np.where(obs_space_1==3)[0][0]
                goal_y = np.where(obs_space_1==3)[1][0]
                obs_space_1[goal_x][goal_y] = 0
                    
            # also remove the wall just because the VAE works better like this.
            if np.where(obs_space_1==4)[0].size != 0:
                goal_x = np.where(obs_space_1==4)[0][0]
                goal_y = np.where(obs_space_1==4)[1][0]
                obs_space_1[goal_x][goal_y] = 0
            
            observation = [obs_space_1, torch.tensor([0, 0])]
            
            view, dir_mag = observation
            view = view.flatten()
            observation = np.concatenate((view, dir_mag))
            
        else:
            observation = self._get_obs()
            
        
        self.done = self.env_master.done
        terminated = self.done
        
        return observation, reward, terminated, None, info
    
    
    def fracos_step(self, action, next_ob, agent, total_rewards=0):
        """This needs to manage recursively taking actions"""
        # !!! Need to add in tracking of the steps -- unless this is done in the infos.
        ob = tuple(next_ob.cpu().numpy())
        
        if action not in range(agent.action_prims):
            id_actions = tuple(agent.discrete_search_cache[ob][action])
            for id_action in id_actions:
                if id_action != None:
                    pass
                next_ob, total_rewards, termination, truncation, info = self.fracos_step(id_action, env, next_ob, agent, total_rewards=total_rewards)
        else:
            next_ob, reward, termination, truncation, info = self.step(action)
            total_rewards += reward
            next_done = np.logical_or(termination, truncation)
            if next_done:
                return next_ob, total_rewards, termination, truncation, info
            
        return next_ob, total_rewards, termination, truncation, info
        
    # def render(self):
    #     if self.render_mode == "rgb_array":
    #         return "rendering has not been implemented yet"
        
    def close(self):
        pass
    
if __name__ == "__main__":
    
    register( id="x4nno/metagrid-v0",
             entry_point="metagrid_gymnasium_wrapper:MetaGridEnv",
             max_episode_steps=500,)
    
    env = gym.make("x4nno/metagrid-v0")
