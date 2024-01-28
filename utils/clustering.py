#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 11:17:02 2023

@author: x4nno
"""

import sys
sys.path.append("/home/x4nno/Documents/PhD/MetaGridEnv/MetaGridEnv") # change to your location
import os
sys.path.append("/home/x4nno/Documents/PhD/FRACOs_v6")


import numpy as np
import pickle
# from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import copy

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight

from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import matplotlib as mpl
# from cuml.cluster import hdbscan
import hdbscan
import umap.umap_ as umap
import random
# from umap.parametric_umap import ParametricUMAP

from utils.eval import create_trajectories, create_random_trajectories, create_ind_opt_trajectories, create_even_trajectories
from cycler import cycler
from utils.VAE_creation import transform_obs_0_vae
from utils.VAE_CNN_creation import get_model
import torch
from torch import nn

import gym
import pickle
from utils.compression import cluster_PI_compression
from utils.VAE_CNN_creation import get_model

from collections import Counter


import MetaGridEnv
from gym.envs.registration import register 

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    #torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
    

colors = ["#EBAC23",
            "#B80058",
            "#008CF9",
            "#006E00",
            "#00BBAD",
            "#D163E6",
            "#B24502",
            "#FF9287",
            "#5954D6",
            "#00C6F8",
            "#878500",
            "#00A76C",
            "#F6DA9C",
            "#FF5CAA",
            "#8ACCFF",
            "#4BFF4B",
            "#6EFFF4",
            "#EDC1F5",
            "#FEAE7C",
            "#FFC8C3",
            "#BDBBEF",
            "#BDF2FF",
            "#FFFC43",
            "#65FFC8",
            "#AAAAAA"]

def show_from_state(state):
    plt.imshow(state[:49].reshape(7,7))
    
def vis_all_envs(env_name):
    env_dir = "trajectories/opt" + f"/{env_name}/0/envs/"
    for f in os.listdir(env_dir):
        fd = os.path.join(env_dir, f)
        with open(fd, "rb") as file:
            env_list = pickle.load(file)
            for env in env_list:
                plt.imshow(env.env_master.domain)
                plt.show()

def create_fractures(trajectories, env_name, obs_space=False, vae_path="pretrained/VAE/MetaGridEnv/vae_cnn.torch",
                     chain_length=2, a_pre_enc=True, fracos_agent=None):
    
    if vae_path != None:
        if  "MetaGridEnv" in env_name:
            model = get_model(vae_path)
            encoder = model.encode
            decoder = model.decode
            
            all_fractures = []
            corre_traj = []
            for trajectory in trajectories:
                move_count=0
                for move in trajectory:
                    state_count = 0
                    for state in move:
                        if torch.is_tensor(state):
                            trajectory[move_count][state_count] = state.cpu().detach()
                        state_count += 1
                    move_count += 1
                    
                trajectory = np.array(trajectory, dtype=object)
                states = trajectory[:,0]
                obs_0_list = []
                obs_0_noenc_list = []
                obs_1_list = []
                for state in states:
                    state = torch.tensor(state).to(device)
                    obs_0 = state[:49]
                    obs_0_noenc_list.append(obs_0.detach().cpu())
                    obs_1 = state[49:]
                    obs_0_enc = encoder(obs_0.float().reshape(1,1,7,7)) # needs to be a float - despite saying double
                    obs_0_list.append(obs_0_enc[0][0].detach().cpu())
                    obs_1_list.append(obs_1.detach().cpu())
                
                obs_0_arr = np.vstack(obs_0_list)
                obs_0_noenc_arr = np.vstack(obs_0_noenc_list)
                
                obs_1_arr = np.vstack(obs_1_list)
                
                obs = np.concatenate((obs_0_arr, obs_1_arr), axis=1)
                obs_no_enc = np.concatenate((obs_0_noenc_arr, obs_1_arr), axis=1)
                
                obs = obs[:-(chain_length)]
                obs_no_enc = obs_no_enc[:-(chain_length)]
                
                actions1 = trajectory[:-(chain_length),1]
                actions1 = np.asarray(actions1)
                actions1 = np.stack(actions1)
                
                frac = np.concatenate((obs, actions1), axis=1)
                frac_noenc = np.concatenate((obs_no_enc, actions1), axis=1)
                
                for b in range(1,chain_length):
                    n_actions = trajectory[b:-(chain_length-b),1]
                    n_actions = np.asarray(n_actions)
                    n_actions = np.stack(n_actions)
                    frac =  np.concatenate((frac, n_actions), axis=1)
                    frac_noenc = np.concatenate((frac_noenc, n_actions), axis=1)
                
                frac = frac.tolist()
                frac_noenc = frac_noenc.tolist()
                
                corre_traj.append(frac_noenc)
                all_fractures.append(frac)
        
        else:
            # !!! PROBABLY THIS IS WRONG NEED TO CHECK THIS BETTER WHEN GET TO PROCGEN
            model = get_model(vae_path)
            encoder = model.encode
            decoder = model.decode
            
            all_fractures = []
            corre_traj = []
            for trajectory in trajectories:
                move_count=0
                for move in trajectory:
                    state_count = 0
                    for state in move:
                        if torch.is_tensor(state):
                            trajectory[move_count][state_count] = state.cpu().detach()
                        state_count += 1
                    move_count += 1
                    
                trajectory = np.array(trajectory, dtype=object)
                states = trajectory[:-1,0]
                state_list = []
                for state in states:
                    state = torch.tensor(state).to(device)
                    # What should be the reshape here?
                    state_enc = encoder(state.float()) # needs to be a float - despite saying double
                    state_list.append(state_enc.detach().cpu())
                    
                state_list_arr = np.vstack(state_list)
                
                if (chain_length-2) == 0:
                    pass
                else:
                    obs = state_list_arr[:-(chain_length-2)]
                
                actions1 = trajectory[:-(chain_length-1),1]
                if a_pre_enc:
                    actions1 = fracos_agent.cypher[int(actions1)]
                actions1 = np.asarray(actions1)
                actions1 = np.stack(actions1)
                
                frac = np.concatenate((obs, actions1), axis=1)
                
                for b in range(1,chain_length):
                    if (chain_length-1-b) == 0:
                        n_actions = trajectory[b:,1]
                    else:
                        n_actions = trajectory[b:-(chain_length-1-b),1]
                        
                    if a_pre_enc:
                        n_actions = fracos_agent.cypher[int(n_actions)]
                    n_actions = np.asarray(n_actions)
                    n_actions = np.stack(n_actions)
                    frac =  np.concatenate((frac, n_actions), axis=1)
                
                frac = frac.tolist()
                
                all_fractures.append(frac)
            
    elif (vae_path == None) and ("MetaGridEnv" in env_name):

        all_fractures = []
        corre_traj = []
        for trajectory in trajectories:
            try:
                move_count=0
                for move in trajectory:
                    state_count = 0
                    for state in move:
                        if torch.is_tensor(state):
                            trajectory[move_count][state_count] = state.cpu().detach()
                        state_count += 1
                    move_count += 1
                    
                trajectory = np.array(trajectory, dtype=object)
                states = trajectory[:,0]
                obs_0_list = []
                obs_0_noenc_list = []
                obs_1_list = []
                for state in states:
                    obs_0 = state[:49]
                    obs_1 = state[49:]
                    obs_0_list.append(obs_0)
                    obs_1_list.append(obs_1)
                
                obs_0_arr = np.vstack(obs_0_list)
                obs_1_arr = np.vstack(obs_1_list)
                
                obs = np.concatenate((obs_0_arr, obs_1_arr), axis=1)
                
                obs = obs[:-(chain_length)]
                
                
                actions1 = trajectory[:-(chain_length),1]
                if a_pre_enc:
                    actions1 = fracos_agent.cypher[int(actions1)]
                actions1 = np.asarray(actions1)
                actions1 = np.stack(actions1)
                
                frac = np.concatenate((obs, actions1), axis=1)
                
                for b in range(1,chain_length):
                    n_actions = trajectory[b:-(chain_length-b),1]
                        
                    if a_pre_enc:
                        n_actions = fracos_agent.cypher[int(n_actions)]
                    n_actions = np.asarray(n_actions)
                    n_actions = np.stack(n_actions)
                    frac =  np.concatenate((frac, n_actions), axis=1)
                
                frac = frac.tolist()
                
                all_fractures.append(frac)
            except:
                print("A trajectory has been found which is shorter than chain length, please reduce chain length")
        
    else:
        
        #!!! Probably this is wrong -- check this when doing a simple method.
        all_fractures = []
        corre_traj = []
        for trajectory in trajectories:
            move_count=0
            for move in trajectory:
                state_count = 0
                for state in move:
                    if torch.is_tensor(state):
                        trajectory[move_count][state_count] = state.cpu().detach()
                    state_count += 1
                move_count += 1
                
            trajectory = np.array(trajectory, dtype=object)
            states = trajectory[:,0]
            state_list = []
            for state in states:
                state_list.append(state)
                
            state_list_arr = np.vstack(state_list)
            
            obs = state_list_arr[:-(chain_length)]
            
            
            actions1 = trajectory[:-(chain_length),1]
            if a_pre_enc:
                actions1 = fracos_agent.cypher[int(actions1)]
            actions1 = np.asarray(actions1)
            actions1 = np.stack(actions1)
            
            frac = np.concatenate((obs, actions1), axis=1)
            
            for b in range(1,chain_length):
                n_actions = trajectory[b:-(chain_length-b),1]
                    
                if a_pre_enc:
                    n_actions = fracos_agent.cypher[int(n_actions)]
                n_actions = np.asarray(n_actions)
                n_actions = np.stack(n_actions)
                frac =  np.concatenate((frac, n_actions), axis=1)
            
            frac = frac.tolist()
            
            all_fractures.append(frac)
                
    # all_fractures is now a list of all fractures in all the trajectories
            
    return all_fractures, corre_traj


def get_all_s_f_index(all_fractures, ep_rewards, failure_std_threshold,
                      use_std=True, failure_min=None):
    
    """Returns the a list of 1 for success and 0 for failure for every fracture """
    
    rew_mean = np.mean(ep_rewards) 
    rew_std = np.std(ep_rewards)
    
    if use_std:
        failure_threshold = rew_mean - failure_std_threshold*rew_std
    else:
        failure_threshold = failure_min
    
    failure_indexes = np.where(np.asarray(ep_rewards) < failure_threshold)[0]
    
    all_s_f = []
    for i in range(len(all_fractures)):
        for j in all_fractures[i]:
            if i in failure_indexes:
                all_s_f.append(0)
            else:
                all_s_f.append(1)
                
    return all_s_f



def tSNE_reduction(X, verbose=False):
    
    """Takes a dataset X and returns a tsne reduction"""
    
    # t_sne = TSNE(n_components=2, learning_rate='auto', init='random', verbose=verbose)
    # X = np.array(X) # because it's in a list and now we have an array of size (list_len, stack_size)
    # X_embedded = t_sne.fit_transform(X)
    # return X_embedded, t_sne
    print("TSNE IS DEPRECATED - USE UMAP")
    pass

def umap_reduction(X, parametric=False):
    
    if parametric:
        print("ERROR THIS HAS NOT BEEN IMPLEMENTED CORRECTLY YET!")
        # reducer = ParametricUMAP(random_state=42, low_memory=True,
        #                          unique=True,
        #                          # parametric_reconstruction= True,
        #                          )
    
        X = np.array(X)
        
    else:
        # print("ERROR have temp removed umap functions")
        reducer = umap.UMAP(random_state=42, n_neighbors=20, min_dist=0.15)
        # reducer = None

    embedding = reducer.fit_transform(X)
    return embedding, reducer

def produce_umap_reducer_pipeline(method, opt_num_of_trajs, rand_num_of_trajs, env,
                                  max_ep_length,
                                  model_path, umap_save_location, env_name,
                                  plotit=False, shuffle=True, max_train_steps=1e4,
                                  MAX_DEPTH=1, CURRENT_DEPTH=0, saved_traj_dir = None,
                                  create_random_trajs = True, failure_min=4, evalu=True,
                                  vae_path="pretrained/VAE/MetaGridEnv/vae_cnn.torch"):
    
    # Typically you want the rand_num_of_trajs to scale with how long the episode 
    # is gonna be go random - on a maze you need less random,
    # but on cartpole you want it longer - as it will fail earlier
    
    if saved_traj_dir is not None:
        optimal_ep_rewards = []
        optimal_trajectories = []
        for file in os.listdir(saved_traj_dir + "/rew"): # rew and traj are the same
            rew_path = saved_traj_dir + "/rew/" + file
            traj_path = saved_traj_dir + "/traj/" + file
            rew_content = pickle.load(open(rew_path, "rb"))
            traj_content = pickle.load(open(traj_path, "rb"))
            
            optimal_ep_rewards += rew_content
            optimal_trajectories += traj_content
            
    
    else:    
        if shuffle:
            optimal_trajectories, optimal_ep_rewards = create_trajectories(method, opt_num_of_trajs,
                                                           model_path, env, max_ep_length,
                                                           obs_space=False, evalu=evalu)
        else:
            optimal_trajectories, optimal_ep_rewards = create_ind_opt_trajectories(method, opt_num_of_trajs,
                                                                          env_name, max_ep_length,
                                                                          max_train_steps,
                                                                          MAX_DEPTH, CURRENT_DEPTH, verbose=False)  
    
    if create_random_trajs:
        random_trajectories, random_ep_rewards = create_random_trajectories(method, rand_num_of_trajs,
                                                       model_path, env, max_ep_length,
                                                       obs_space=False)
        all_trajectories = optimal_trajectories + random_trajectories
    else:
        random_trajectories = []
        random_ep_rewards = []
        all_trajectories = optimal_trajectories
        
    all_ep_rewards = optimal_ep_rewards + random_ep_rewards
    all_fractures, corre_traj = create_fractures(all_trajectories, env_name, vae_path=vae_path)
    concat_fractures = sum(all_fractures, [])

    embedding, reducer = umap_reduction(concat_fractures)
    
    
    all_s_f = get_all_s_f_index(concat_fractures, all_ep_rewards, failure_std_threshold=None,
                                use_std=False, failure_min=failure_min)
    # for i in range(len(optimal_trajectories)):
    #     for j in range(len(optimal_trajectories[i])):
    #         all_s_f.append(1)
    # for i in range(len(random_trajectories)):
    #     for j in range(len(random_trajectories[i])):
    #         all_s_f.append(0)

    if plotit:
        plot_umap(embedding, all_s_f, env_name)
        
    return embedding, reducer, all_ep_rewards, all_trajectories, all_s_f

def produce_umap_reducer_pipeline_multi(method, checkpoints, env,
                                        umap_save_location, env_name, labels, even=True,
                                        vae_path=None, num_of_trajs=None, max_timesteps=20000,
                                        rew_gen_meta=True, evalu=True, chain_length=2, include_rand_agent=True,
                                        one_level=True, offline_path=None, 
                                        min_traj_len=None):
     
    colors = [
    'blue', 'orange', 'green', 'red', 'purple',
    'brown', 'pink', 'gray', 'olive', 'cyan',
    'lime', 'teal', 'indigo', 'maroon', 'gold',
    'navy', 'salmon', 'darkgreen', 'orchid', 'sienna'
    ]
    
    list_of_trajectories = []
    list_of_rewards = []
    list_of_fractures = []
    list_of_concat_fractures = []
    
    if offline_path is not None:
        for file in os.listdir(offline_path + "/rew"): # rew and traj are the same
            if file == "bash.sh":
                pass
            else:
                rew_path = offline_path + "/rew/" + file
                traj_path = offline_path + "/traj/" + file
                rew_content = pickle.load(open(rew_path, "rb"))
                traj_content = pickle.load(open(traj_path, "rb"))
                
                if min_traj_len:
                    new_traj_content = []
                    for i in range(len(traj_content)):
                        if len(traj_content[i]) >= min_traj_len:
                            new_traj_content.append(traj_content[i])
                    traj_content = new_traj_content
                
                list_of_trajectories.append(traj_content)
                list_of_rewards.append(rew_content)

                all_fractures, corre_traj = create_fractures(traj_content, env_name,
                                                             vae_path=vae_path, fracos_agent=method,
                                                             a_pre_enc=False, chain_length=chain_length)
                concat_fractures = sum(all_fractures, [])
                
                list_of_concat_fractures.append(concat_fractures)
                list_of_fractures.append(all_fractures)
        
        print("loaded trajectories")
        labels = ["Optimal"]
        
    else:
        
        ## no trained 
        start_env = copy.deepcopy(env)
        plt.imshow(env.env_master.domain)
        plt.savefig("umap_images/{}_one_level_domain.svg".format(env_name), format="svg")
        plt.clf()
        
        if include_rand_agent:
            optimal_trajectories, optimal_ep_rewards = create_even_trajectories(method, max_timesteps,
                                                           None, start_env, max_ep_length=250,
                                                           obs_space=False, evalu=evalu, rew_gen_meta=rew_gen_meta,
                                                           one_level=one_level)
            
            list_of_trajectories.append(optimal_trajectories)
            list_of_rewards.append(optimal_ep_rewards)
    
            all_fractures, corre_traj = create_fractures(optimal_trajectories, env_name,
                                                         vae_path=vae_path, fracos_agent=method,
                                                         a_pre_enc=False, chain_length=chain_length)
            concat_fractures = sum(all_fractures, [])
            
            list_of_concat_fractures.append(concat_fractures)
            list_of_fractures.append(all_fractures)
            
            labels.insert(0,0)
            
        # start with trained
        for model_path in checkpoints:
            start_env = copy.deepcopy(env)
            plt.imshow(start_env.env_master.domain)
            plt.show()
            if even:
                optimal_trajectories, optimal_ep_rewards = create_even_trajectories(method, max_timesteps,
                                                               model_path, start_env, max_ep_length=250,
                                                               obs_space=False, evalu=evalu, rew_gen_meta=rew_gen_meta,
                                                               one_level=one_level)
            else:
                optimal_trajectories, optimal_ep_rewards = create_ind_opt_trajectories(method, num_of_trajs,
                                                                              env_name, max_ep_length=250,
                                                                              max_train_steps=1000,
                                                                              MAX_DEPTH=3, CURRENT_DEPTH=0, verbose=False)
        
            list_of_trajectories.append(optimal_trajectories)
            list_of_rewards.append(optimal_ep_rewards)
    
            all_fractures, corre_traj = create_fractures(optimal_trajectories, env_name,
                                                         vae_path=vae_path, fracos_agent=method,
                                                         a_pre_enc=False, chain_length=chain_length)
            concat_fractures = sum(all_fractures, [])
            
            list_of_concat_fractures.append(concat_fractures)
            list_of_fractures.append(all_fractures)
            
            print(model_path + " completed")
        
    pre_umap = sum(list_of_concat_fractures, [])
    embedding, reducer = umap_reduction(pre_umap)
    
    if umap_save_location is not None:
        pickle.dump(embedding, open( umap_save_location, "wb" ))
    
    x = embedding[:,0]
    y = embedding[:,1]
    
    if offline_path is not None:
        plt.scatter(x, y, s=2, label="Optimal")
        plt.title("{} fracture visulisations optimal".format(env_name))
        plt.legend(bbox_to_anchor=(1, 1))
        # plt.xlim(-40,40)
        # plt.ylim(-40,40)
        plt.savefig("umap_images/{}_all_offline_optimal.svg".format(env_name), format="svg")
        plt.show()
    else:
        counter = 0
        for i in range(len(list_of_concat_fractures)):
            embedding_plot_help_x = []
            embedding_plot_help_y = []
            for j in range(len(list_of_concat_fractures[i])):
                embedding_plot_help_x.append(x[counter])
                embedding_plot_help_y.append(y[counter])
                counter += 1
            
            plt.scatter(embedding_plot_help_x, embedding_plot_help_y, s=2, label=(labels[i]), color=colors[i])
        
        plt.title("{} fracture visulisations".format(env_name))
        plt.legend(bbox_to_anchor=(1, 1))
        # plt.xlim(-40,40)
        # plt.ylim(-40,40)
        plt.savefig("umap_images/{}_all.svg".format(env_name), format="svg")
        plt.show()
        plt.clf()
        
        ## Plot each individually too:
        counter = 0
        for i in range(len(list_of_concat_fractures)):
            embedding_plot_help_x = []
            embedding_plot_help_y = []
            for j in range(len(list_of_concat_fractures[i])):
                embedding_plot_help_x.append(x[counter])
                embedding_plot_help_y.append(y[counter])
                counter += 1
            
            plt.scatter(embedding_plot_help_x, embedding_plot_help_y, s=2, label=(labels[i]), color=colors[i])
        
            plt.title("{} fracture visulisations".format(env_name))
            plt.legend(bbox_to_anchor=(1, 1))
            # plt.xlim(-40,40)
            # plt.ylim(-40,40)
            plt.savefig("umap_images/{}_{}.svg".format(env_name, labels[i]), format="svg")
            plt.show()
            plt.clf()
        
    return list_of_trajectories, list_of_rewards, list_of_fractures, embedding, reducer

def plot_umap(umap_embedding, all_s_f, env_name=""):
    """Takes a tSNE reduced data and the all_s_f files and plots."""
    plt.gca().set_prop_cycle(cycler('color', colors))
    
    x = umap_embedding[:,0]
    y = umap_embedding[:,1]

    # to plot these easy let us now seperate to success and failure

    success_x = []
    success_y = []
    failure_x = []
    failure_y = []

    for i in range(len(x)):
        if all_s_f[i] > 0:
            success_x.append(x[i])
            success_y.append(y[i])
        else:
            failure_x.append(x[i])
            failure_y.append(y[i])

    plt.scatter(success_x, success_y, s=1, label = "Success")
    plt.scatter(failure_x, failure_y, s=1, label = "Failure")

    plt.title("{} umap latent space visualisation for t=t obs space and t=t, t+1 actions".format(env_name))
    plt.legend(bbox_to_anchor=(1, 1))
    plt.show()
    plt.clf()
    
    # plot just the success
    
    plt.scatter(success_x, success_y, s=1, label = "Success",
                )
    # plt.scatter(failure_x_tSNE, failure_y_tSNE, c="r", s=1, label = "Failure")

    plt.title("{} umap latent space visualisation for t=t obs space and t=t, t+1 actions".format(env_name))
    plt.legend()
    plt.show()

def plot_tsne(all_tSNE_vars, all_s_f, env_name=""):
    
    """Takes a tSNE reduced data and the all_s_f files and plots."""
    
    x_tSNE = all_tSNE_vars[:,0]
    y_tSNE = all_tSNE_vars[:,1]

    # to plot these easy let us now seperate to success and failure

    success_x_tSNE = []
    success_y_tSNE = []
    failure_x_tSNE = []
    failure_y_tSNE = []

    for i in range(len(x_tSNE)):
        if all_s_f[i] > 0:
            success_x_tSNE.append(x_tSNE[i])
            success_y_tSNE.append(y_tSNE[i])
        else:
            failure_x_tSNE.append(x_tSNE[i])
            failure_y_tSNE.append(y_tSNE[i])

    plt.scatter(success_x_tSNE, success_y_tSNE, c="g", s=1, label = "Success")
    plt.scatter(failure_x_tSNE, failure_y_tSNE, c="r", s=1, label = "Failure")

    plt.title("{} tSNE latent space visualisation for t=t obs space and t=t, t+1 actions".format(env_name))
    plt.legend()
    plt.show()
    
    # plot just the success
    
    plt.scatter(success_x_tSNE, success_y_tSNE, c="g", s=1, label = "Success")
    # plt.scatter(failure_x_tSNE, failure_y_tSNE, c="r", s=1, label = "Failure")

    plt.title("{} tSNE latent space visualisation for t=t obs space and t=t, t+1 actions".format(env_name))
    plt.legend()
    plt.show()
    
def create_clusterer(all_fractures, MIN_CLUSTER_SIZE=30, metric="euclidean"):
    # WE MAY want to use umap as the reduction technique to cluster!?
    # instead of a VAE?
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, gen_min_span_tree=True,
                                prediction_data=True, metric=metric)
    
    if not isinstance(all_fractures, np.ndarray): # as will be a list of lists
        concat_fractures = sum(all_fractures, [])
        concat_fractures = np.asarray(concat_fractures)
    
    else:
        concat_fractures = all_fractures
    
    clusterer.fit(concat_fractures)
    
    return clusterer

def create_clusterer_parametric(all_fractures, MIN_CLUSTER_SIZE=30):
    # WE MAY want to use umap as the reduction technique to cluster!?
    # instead of a VAE?
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, gen_min_span_tree=True,
                                prediction_data=True)
    
    if not isinstance(all_fractures, np.ndarray): # as will be a list of lists
        concat_fractures = sum(all_fractures, [])
        concat_fractures = np.asarray(concat_fractures)
    
    else:
        concat_fractures = all_fractures
    
    clusterer.fit(concat_fractures)
    
    return clusterer


def colour_clusters_on_UMAP(umap_embedding, clusterer, env_name, legend=True):
    """Takes embeddings from a UMAP reduction and plots them 
    coloured by which cluster they are in"""
    
    plt.gca().set_prop_cycle(cycler('color', colors))
    
    x = umap_embedding[:,0]
    y = umap_embedding[:,1]
    
    labels, strengths = hdbscan.approximate_predict(clusterer, umap_embedding)
    
    
    ##
    
    ##
    print_dict_x = {}
    print_dict_y = {}
    for uni_label in np.unique(labels):
        print_dict_x[uni_label] = []
        print_dict_y[uni_label] = []
        for i in range(len(umap_embedding)):
            if labels[i] == uni_label:
                print_dict_x[uni_label].append(x[i])
                print_dict_y[uni_label].append(y[i])
                
    
    for cluster in print_dict_x.keys(): # x and y are the same
        if cluster == -1:
            plt.scatter(print_dict_x[cluster],
                        print_dict_y[cluster], s=2, label=cluster, c = "black")
        else:   
            plt.scatter(print_dict_x[cluster],
                        print_dict_y[cluster], s=2, label=cluster)
    
    plt.title("UMAP showing discovered clusters")
    
            
    plt.savefig("umap_images/{}_clusters.svg".format(env_name), format="svg", bbox_inches='tight')
    
    if legend:
        lgnd = plt.legend(bbox_to_anchor=(1, 1))
        for i in range(len(lgnd.legendHandles)):
            lgnd.legendHandles[i]._sizes = [30]
            
    plt.savefig("umap_images/{}_clusters_labels.svg".format(env_name), format="svg", bbox_inches='tight')
    plt.clf()



def find_goal_from_dirdis(direction, distance):
    # print(direction, distance)
    
    if 0 < direction <= np.pi/2: # top right
        x_ref = distance*np.cos(direction)
        y_ref = -distance*np.sin(direction)
    elif np.pi/2 < direction <= np.pi: # top left
        x_ref = -distance*np.cos(np.pi - direction)
        y_ref = -distance*np.sin(np.pi-direction)
    elif -np.pi < direction <= -np.pi/2: # bottom left
        x_ref = -distance*np.cos(np.pi+direction)
        y_ref = distance*np.sin(np.pi+direction)
    elif -np.pi/2 < direction <= 0: # bottom right
        x_ref = distance*np.cos(-direction)
        y_ref = distance*np.sin(-direction)
        
    return x_ref, y_ref

def visualize_clusters(concat_fractures, concat_trajs, clusterer, cluster_to_vis, method, average=False,
                       samples=10, env_name="MetaGridEnv", vae=True, MAX_DEPTH=3,
                       chain_length=2, from_offline_prim=False):
    if "MetaGridEnv" not in env_name:
        print("""This function currently only works for MetaGridEnv - to use it for 
              other environments you will need to change this function""")
        return    
    
    labels = clusterer.labels_
    indexes = np.where(labels == cluster_to_vis)[0]
    # print(indexes)
    all_traj_to_cluster = []
    true_traj_to_cluster = []
    for index in indexes:
        all_traj_to_cluster.append(concat_fractures[index])
        if vae:
            true_traj_to_cluster.append(concat_trajs[index])
    
    # Either average or produce several visualisations
    if average:
        print("not implemented yet")
    else:
        figure, axis = plt.subplots(2, samples)
        for i in range(samples):
            random_idx = random.choice(range(len(all_traj_to_cluster)))
            random_traj = all_traj_to_cluster[random_idx]
            if vae:
                random_true_traj = true_traj_to_cluster[random_idx]
            # print(random_traj)
            if not from_offline_prim:
                if vae:
                    state = method.decoder(torch.tensor(random_traj[:4]).view(1,4).to(device))
                    state = state.reshape((7,7))
                    state[3,3] = 2 # agent_loc
                    state = state.cpu().detach().numpy()
                    goal_location = random_traj[4:6]
                    action_list = []
                    for a in range(chain_length):
                        # print(6+a+a*MAX_DEPTH)
                        # print(7+a+(a+1)*MAX_DEPTH)
                        # print("------")
                        acti = random_traj[6+a*(MAX_DEPTH*method.max_clusters_per_clusterer+method.action_dim): \
                                           6+(a+1)*(MAX_DEPTH*method.max_clusters_per_clusterer+method.action_dim)]
                        # print(6+a*(MAX_DEPTH*method.max_clusters_per_clusterer+method.action_dim))
                        # print(6+a+(a+1)*(MAX_DEPTH*method.max_clusters_per_clusterer+method.action_dim))
                        action_list.append(method.reverse_cypher[tuple(acti)])
                else:
                    #print("NOT IMPLEMENTED CORRECTLY YET")
                    state = np.asarray(random_traj[:49])
                    state = state.reshape((7,7))
                    state[3,3] = 2 # agent_loc
                    goal_location = random_traj[49:51]
                    action_list = []
                    for a in range(chain_length):
                        # print(6+a+a*MAX_DEPTH)
                        # print(7+a+(a+1)*MAX_DEPTH)
                        # print("------")
                        acti = random_traj[51+a*(MAX_DEPTH*method.max_clusters_per_clusterer+method.action_dim): \
                                           51+(a+1)*(MAX_DEPTH*method.max_clusters_per_clusterer+method.action_dim)]
                        # print(6+a*(MAX_DEPTH*method.max_clusters_per_clusterer+method.action_dim))
                        # print(6+a+(a+1)*(MAX_DEPTH*method.max_clusters_per_clusterer+method.action_dim))
                        action_list.append(method.reverse_cypher[tuple(acti)])
                
            else:
                # Will only work for one level. Will need to change for two levels.
                method.clusterers = []
                method.clusters = []
                method.cypher, method.reverse_cypher = method.gen_cypher(method.current_depth-1)
                method.get_clusters(env_name)
                if vae:
                    state = method.decoder(torch.tensor(random_traj[:4]).view(1,4).to(device))
                    state = state.reshape((7,7))
                    state[3,3] = 2 # agent_loc
                    state = state.cpu().detach().numpy()
                    goal_location = random_traj[4:6]
                    action_list = []
                    for a in range(chain_length):
                        # print(6+a+a*MAX_DEPTH)
                        # print(7+a+(a+1)*MAX_DEPTH)
                        # print("------")
                        acti = random_traj[6+a*(method.current_depth*method.max_clusters_per_clusterer+method.action_dim): \
                                           6+(a+1)*(method.current_depth*method.max_clusters_per_clusterer+method.action_dim)]
                        # print(6+a*(MAX_DEPTH*method.max_clusters_per_clusterer+method.action_dim))
                        # print(6+a+(a+1)*(MAX_DEPTH*method.max_clusters_per_clusterer+method.action_dim))
                        action_list.append(method.reverse_cypher[tuple(acti)])
                else:
                    #print("NOT IMPLEMENTED CORRECTLY YET")
                    state = np.asarray(random_traj[:49])
                    state = state.reshape((7,7))
                    state[3,3] = 2 # agent_loc
                    goal_location = random_traj[49:51]
                    action_list = []
                    for a in range(chain_length):
                        # print(6+a+a*MAX_DEPTH)
                        # print(7+a+(a+1)*MAX_DEPTH)
                        # print("------")
                        acti = random_traj[51+a*((method.current_depth-1)*method.max_clusters_per_clusterer+method.action_dim): \
                                           51+(a+1)*((method.current_depth-1)*method.max_clusters_per_clusterer+method.action_dim)]
                        # print(6+a*(MAX_DEPTH*method.max_clusters_per_clusterer+method.action_dim))
                        # print(6+a+(a+1)*(MAX_DEPTH*method.max_clusters_per_clusterer+method.action_dim))
                        action_list.append(method.reverse_cypher[tuple(acti)])
                
            x_ref, y_ref = find_goal_from_dirdis(goal_location[0], goal_location[1])
            x_ref = int(x_ref)+3
            y_ref = int(y_ref)+3
            # print(state)
            # if (x_ref <= 3) and (y_ref <= 3): # keep within image
            #     state[x_ref, y_ref] = 3 # goal loc
            
            goal_location = [ '%.2f' % elem for elem in goal_location]
            
            axis[0, i].imshow(state, cmap='Blues')
            axis[0, i].set_title("{}, {}".format(goal_location, action_list), fontsize=6)
            # axis[0, i].set_xticks([])
            # axis[0, i].set_yticks([])            
            
            if vae:
                true_state = np.asarray(random_true_traj[0][:49]).reshape((7,7))
                true_state[3,3] = 2 # agent_loc
                true_goal_location = random_true_traj[0][49:51]
                true_goal_location = [ '%.2f' % elem for elem in true_goal_location]
                
                axis[1, i].imshow(true_state, cmap='Blues')
                axis[1, i].set_title("{}, {}".format(true_goal_location, action_list), fontsize=6)
                # axis[0, i].set_xticks([])
                # axis[0, i].set_yticks([])   

        plt.setp(axis, xticks=[], yticks=[])
        figure.suptitle("Cluster {}".format(cluster_to_vis))
        #plt.colorbar()
        plt.show()
        plt.clf()
        
def visualize_cluster_one(all_traj_to_cluster, concat_trajs, clusterer, cluster_to_vis,
                          method, sample, average=False, env_name="MetaGridEnv", vae=True, MAX_DEPTH=3,
                       chain_length=2):
    if env_name != "MetaGridEnv":
        print("""This function currently only works for MetaGridEnv - to use it for 
              other environments you will need to change this function""")
        return    
    

    
    # Either average or produce several visualisations
    if average:
        print("not implemented yet")
    else:
        random_traj = all_traj_to_cluster[sample]
        # print(random_traj)
        if vae:
            state = method.decoder(torch.tensor(random_traj[:4]).view(1,4).to(device))
            state = state.reshape((7,7))
            state[3,3] = 2 # agent_loc
            state = state.cpu().detach().numpy()
            goal_location = random_traj[4:6]
            action_list = []
            for a in range(chain_length):
                # print(6+a+a*MAX_DEPTH)
                # print(7+a+(a+1)*MAX_DEPTH)
                # print("------")
                acti = random_traj[6+a*(MAX_DEPTH*method.max_clusters_per_clusterer+method.action_dim): \
                                   6+(a+1)*(MAX_DEPTH*method.max_clusters_per_clusterer+method.action_dim)]
                # print(6+a*(MAX_DEPTH*method.max_clusters_per_clusterer+method.action_dim))
                # print(6+a+(a+1)*(MAX_DEPTH*method.max_clusters_per_clusterer+method.action_dim))
                action_list.append(method.reverse_cypher[tuple(acti)])
        else:
            #print("NOT IMPLEMENTED CORRECTLY YET")
            state = np.asarray(random_traj[:49])
            state = state.reshape((7,7))
            state[3,3] = 2 # agent_loc
            goal_location = random_traj[49:51]
            action_list = []
            for a in range(chain_length):
                # print(6+a+a*MAX_DEPTH)
                # print(7+a+(a+1)*MAX_DEPTH)
                # print("------")
                acti = random_traj[51+a*(MAX_DEPTH*method.max_clusters_per_clusterer+method.action_dim): \
                                   51+(a+1)*(MAX_DEPTH*method.max_clusters_per_clusterer+method.action_dim)]
                # print(6+a*(MAX_DEPTH*method.max_clusters_per_clusterer+method.action_dim))
                # print(6+a+(a+1)*(MAX_DEPTH*method.max_clusters_per_clusterer+method.action_dim))
                action_list.append(method.reverse_cypher[tuple(acti)])
            
        x_ref, y_ref = find_goal_from_dirdis(goal_location[0], goal_location[1])
        x_ref = int(x_ref)+3
        y_ref = int(y_ref)+3
        # print(state)
        # if (x_ref <= 3) and (y_ref <= 3): # keep within image
        #     state[x_ref, y_ref] = 3 # goal loc
        
        goal_location = [ '%.2f' % elem for elem in goal_location]
        
        plt.imshow(state, cmap='Blues')
        plt.title("{}, {}".format(goal_location, action_list))
        plt.show()
        plt.clf()
            
def act_no_to_clstr(act_no, fracos_agent):
    "Only for MetaGridEnv"
    count = 4 # start 4 for primitive
    clstrs_count = -1
    if act_no < 4:
        return clstrs_count, act_no
    clstrs_count = 0
    for clstrs in fracos_agent.clusters:
        if act_no < count + len(clstrs):
            return clstrs, clstrs[act_no-count]
        else:
            count = count + len(clstrs)
        clstrs_count += 1
        
def visualize_clusters_deep(depth, cluster_to_vis, method,
                            average=False, samples=10, env_name="MetaGridEnv",
                            vae=True, MAX_DEPTH=3, chain_length=2, from_offline_prim=False,
                            ):
    
    if "MetaGridEnv" not in env_name:
        print("""This function currently only works for MetaGridEnv - to use it for 
              other environments you will need to change this function""")
        return    
    
    cluster_dir="fracos_clusters/"
    
    clusterer = pickle.load(open(cluster_dir + env_name + "/clusterers/" + "clusterer{}.p".format(depth-1), "rb"))
    concat_fractures = pickle.load(open(cluster_dir + env_name + "/other/" + "concat_fractures{}.p".format(depth-1), "rb"))
    concat_trajs = pickle.load(open(cluster_dir + env_name + "/other/" + "concat_trajs{}.p".format(depth-1), "rb"))
    # cyphers
    cypher = pickle.load(open("fracos_clusters/MetaGridEnv/cluster_cyphers/"+ "cypher_{}.p".format(depth-1), "rb"))
    reverse_cypher = pickle.load(open("fracos_clusters/MetaGridEnv/cluster_reverse_cyphers/" + "cypher_{}.p".format(depth-1), "rb"))
    
    labels = clusterer.labels_
    indexes = np.where(labels == cluster_to_vis)[0]
    # print(indexes)
    all_traj_to_cluster = []
    true_traj_to_cluster = []
    for index in indexes:
        all_traj_to_cluster.append(concat_fractures[index])
        if vae:
            true_traj_to_cluster.append(concat_trajs[index])
    
    # Either average or produce several visualisations
    if average:
        print("not implemented yet")
    else:
        figure, axis = plt.subplots(2, samples)
        for i in range(samples):
            random_idx = random.choice(range(len(all_traj_to_cluster)))
            random_traj = all_traj_to_cluster[random_idx]
            if vae:
                random_true_traj = true_traj_to_cluster[random_idx]
            # print(random_traj)
            if not from_offline_prim:
                if vae:
                    state = method.decoder(torch.tensor(random_traj[:4]).view(1,4).to(device))
                    state = state.reshape((7,7))
                    state[3,3] = 2 # agent_loc
                    state = state.cpu().detach().numpy()
                    goal_location = random_traj[4:6]
                    action_list = []
                    for a in range(chain_length):
                        # print(6+a+a*MAX_DEPTH)
                        # print(7+a+(a+1)*MAX_DEPTH)
                        # print("------")
                        acti = random_traj[6+a*(MAX_DEPTH*method.max_clusters_per_clusterer+method.action_dim): \
                                           6+(a+1)*(MAX_DEPTH*method.max_clusters_per_clusterer+method.action_dim)]
                        # print(6+a*(MAX_DEPTH*method.max_clusters_per_clusterer+method.action_dim))
                        # print(6+a+(a+1)*(MAX_DEPTH*method.max_clusters_per_clusterer+method.action_dim))
                        action_list.append(reverse_cypher[tuple(acti)])
                else:
                    #print("NOT IMPLEMENTED CORRECTLY YET")
                    state = np.asarray(random_traj[:49])
                    state = state.reshape((7,7))
                    state[3,3] = 2 # agent_loc
                    goal_location = random_traj[49:51]
                    all_actions = random_traj[51:]
                    even_splits = len(all_actions) / chain_length
                    
                    pre_action_list = [all_actions[i*even_splits: (i+1) * even_splits] for i in range(chain_length)]
                    action_list = [reverse_cypher[tuple(ac)] for ac in pre_action_list]
                
            else:
                # Will only work for one level. Will need to change for two levels.
                if vae:
                    state = method.decoder(torch.tensor(random_traj[:4]).view(1,4).to(device))
                    state = state.reshape((7,7))
                    state[3,3] = 2 # agent_loc
                    state = state.cpu().detach().numpy()
                    goal_location = random_traj[4:6]
                    all_actions = random_traj[51:]
                    even_splits = len(all_actions) / chain_length
                    pre_action_list = [all_actions[i*even_splits: (i+1) * even_splits] for i in range(chain_length)]
                    action_list = [reverse_cypher[tuple(ac)] for ac in pre_action_list]
                else:
                    state = np.asarray(random_traj[:49])
                    state = state.reshape((7,7))
                    state[3,3] = 2 # agent_loc
                    goal_location = random_traj[49:51]
                    all_actions = random_traj[51:]
                    even_splits = len(all_actions) // chain_length 
                    pre_action_list = [all_actions[i*even_splits: (i+1) * even_splits] for i in range(chain_length)]
                    action_list = [reverse_cypher[tuple(ac)] for ac in pre_action_list]
                    
            x_ref, y_ref = find_goal_from_dirdis(goal_location[0], goal_location[1])
            x_ref = int(x_ref)+3
            y_ref = int(y_ref)+3
            # print(state)
            # if (x_ref <= 3) and (y_ref <= 3): # keep within image
            #     state[x_ref, y_ref] = 3 # goal loc
            
            goal_location = [ '%.2f' % elem for elem in goal_location]
            
            axis[0, i].imshow(state, cmap='Blues')
            axis[0, i].set_title("{}, {}".format(goal_location, action_list), fontsize=6)
            # axis[0, i].set_xticks([])
            # axis[0, i].set_yticks([])            
            
            if vae:
                true_state = np.asarray(random_true_traj[0][:49]).reshape((7,7))
                true_state[3,3] = 2 # agent_loc
                true_goal_location = random_true_traj[0][49:51]
                true_goal_location = [ '%.2f' % elem for elem in true_goal_location]
                
                axis[1, i].imshow(true_state, cmap='Blues')
                axis[1, i].set_title("{}, {}".format(true_goal_location, action_list), fontsize=6)
                # axis[0, i].set_xticks([])
                # axis[0, i].set_yticks([])   

        plt.setp(axis, xticks=[], yticks=[])
        figure.suptitle("Cluster {}".format(cluster_to_vis))
        #plt.colorbar()
        plt.show()
        plt.clf()
        
        for a in action_list:
            if a > 3: # then this is not a primitive and we should print out what it is.
                clstr_idx, cluster = act_no_to_clstr(a, method)
                visualize_clusters_deep(clstr_idx, cluster, method)


class torch_classifier(nn.Module):
    def __init__(self, fracture_shape, num_labels):
        super().__init__()
        initial_shape = fracture_shape
        self.layer_1 = nn.Linear(in_features=initial_shape, out_features=256)
        self.layer_2 = nn.Linear(in_features=256, out_features=256)
        self.layer_3 = nn.Linear(in_features=256, out_features=256)
        self.layer_4 = nn.Linear(in_features=256, out_features=num_labels)
        self.relu = nn.ReLU()
    
    # 3. Define a forward method containing the forward pass computation
    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.layer_4(x)
        return x

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc

def supplement_fractures(fractures, labels, clusterer, all_possible_action_combs, env,
                         gen_str = 0.2, supp_amount=50, vae_path=None):
    
    supp_fractures = []
    supp_labels = []
    for idx in range(len(fractures)):
        
        if vae_path:
            # 6 is the latent VAE shape + dir dis. Needs to be changed depending on VAE
            frac_0 = fractures[idx][:6] 
        else:
            try:
                frac_0 = fractures[idx][:env.observation_space.shape[0]]
            except:
                frac_0 = np.array([fractures[idx][0],])
                
        if len(all_possible_action_combs) < supp_amount:
            for apac in all_possible_action_combs:
                first = True
                for a in apac:
                    if first:
                        ac = a
                        first = False
                    else:
                        ac = np.concatenate((ac, a))
                new_frac = np.concatenate((np.asarray(frac_0),ac))
                if list(new_frac) not in fractures:
                    supp_fractures.append(new_frac)
                    supp_labels.append(-1)
                
        else:
            i = 0
            while i < supp_amount:
                apac = random.choice(all_possible_action_combs)
                first = True
                for a in apac:
                    if first:
                        ac = a
                        first = False
                    else:
                        ac = np.concatenate((ac, a))
                new_frac = np.concatenate((np.asarray(frac_0),ac))
                if list(new_frac) not in fractures:
                    supp_fractures.append(new_frac)
                    supp_labels.append(-1)
                    i += 1
                    
        print("{}/{}".format(idx, len(fractures)))
    
    supp_fractures = np.asarray(supp_fractures)
    
    _, strengths = hdbscan.approximate_predict(clusterer, fractures)
    
    all_fractures = np.concatenate((fractures, supp_fractures))
    
    labels = np.concatenate((labels, supp_labels))
    
    # unhash to remove any cluster predictions which don't pass the gen_str barrier.
    
    for i in range(len(strengths)):
        if strengths[i] <= 1-gen_str:
            labels[i] = -1            
        

    
    # print("Oversampling for labels")
    # all_fractures, labels = ros.fit_resample(all_fractures, labels)
    
            
    return all_fractures, labels
        
def find_single_occurrence_numbers_with_indices(lst):
    counts = Counter(lst)
    single_occurrence_data = [(num, index) for index, num in enumerate(lst) if counts[num] == 1]
    return single_occurrence_data


def remove_single_occurrence_numbers(lst):
    sod = find_single_occurrence_numbers_with_indices(lst)
    for s in sod:
        lst[s[1]] = -1
    
    return lst

def create_NN_clusters(fractures, labels, clusterer, all_possible_action_combs, env,
                       verbose=False, max_epochs=30000, gen_str=0.2, chain_len=2,
                       vae_path=None):
    
    # supplement fractures with 0 labels
    old_fractures = copy.deepcopy(fractures)
    old_labels = copy.deepcopy(labels)
    
    eval_X = torch.tensor(old_fractures).to(device)
    eval_y = torch.tensor(old_labels).type(torch.LongTensor).to(device)
    
    class_weights = None
    
    print("Supplementing Fractures")
    
    fractures, labels = supplement_fractures(fractures, labels, clusterer, env,
                                              all_possible_action_combs, supp_amount=1,
                                              gen_str=gen_str, vae_path=vae_path)
    
    
    fractures = np.asarray(fractures)
    fractures = fractures.astype(float)
    
    # We need to remove any cluster which has only one label?
    unique_labels = set(labels)
    num_labels = len(unique_labels)
    
    # labels = remove_single_occurrence_numbers(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(fractures, 
                                                        labels, 
                                                        stratify=labels,
                                                        random_state=1)
    
    # replace our negative values.
    # At the moment we use 0 and then sacrifice it later by removing it from the 
    # allowed fracos.
    # There must be a better method ... 
    for i in range(len(y_train)):
        if y_train[i] == -1:
            y_train[i] = num_labels-1
    for i in range(len(y_test)):
        if y_test[i] == -1:
            y_test[i] = num_labels-1
    
        
    print("getting weights")
    class_weights=compute_class_weight('balanced',classes=np.unique(y_train),y=y_train)
    class_weights=torch.tensor(class_weights,dtype=torch.float).to(device)
    
    model_args = [len(fractures[0]), len(set(labels))]
    model_0 = torch_classifier(len(fractures[0]), len(set(labels))).to(device)
    
    if class_weights is not None:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights,reduction='mean')
    else:
        loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model_0.parameters(), 
                            lr=0.1)
    
    # Set the number of epochs
    
    # Put data to target device
    X_train, y_train = torch.tensor(X_train).to(device), torch.tensor(y_train).type(torch.LongTensor).to(device)
    X_test, y_test = torch.tensor(X_test).to(device), torch.tensor(y_test).type(torch.LongTensor).to(device)
    
    X_train = X_train.float()
    X_test = X_test.float()
    
    # Build training and evaluation loop
    
    
    last_test_loss = 100000
    for epoch in range(max_epochs):
        # 1. Forward pass
        y_logits = model_0(X_train).squeeze()
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1) # logits -> prediction probabilities -> prediction labels
        
        # 2. Calculate loss and accuracy
        loss = loss_fn(y_logits, y_train) 
        acc = accuracy_fn(y_true=y_train, 
                          y_pred=y_pred)
        
        # 3. Optimizer zero grad
        optimizer.zero_grad()
    
        # 4. Loss backward
        loss.backward()
    
        # 5. Optimizer step
        optimizer.step()
    
        ### Testing
        model_0.eval()
        with torch.inference_mode():
          # 1. Forward pass
          test_logits = model_0(X_test).squeeze()
          test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
          # 2. Calcuate loss and accuracy
          test_loss = loss_fn(test_logits, y_test)
          test_acc = accuracy_fn(y_true=y_test,
                                 y_pred=test_pred)
    
        # Print out what's happening
        if epoch % 100 == 0:
            if verbose:
                print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")
            
    
        # print("Final accuracy on all clusters (ignoring the outliers)")
        # with (torch.inference_mode()) and (torch.no_grad()):
        #     # 1. Forward pass
        #     final_logits = model_0(eval_X).squeeze()
        #     final_pred = torch.softmax(final_logits, dim=1).argmax(dim=1)
        #     # 2. Calcuate loss and accuracy
        #     final_loss = loss_fn(final_logits, eval_y)
        #     final_acc = accuracy_fn(y_true=eval_y,
        #                            y_pred=final_pred)
            
        #     print("Final loss = {}. Final accuracy = {}%".format(final_loss, final_acc))
    
    
    return model_0, model_args

def create_NN_clusters_test(train_fractures, test_fractures, train_labels, test_labels, verbose=False):
    cl_model = create_NN_clusters(train_fractures, train_labels, verbose=verbose)
    return cl_model

def save_all_clusterings(clusterer, clusters, concat_fractures, concat_trajs, NN, model_args, method, cluster_level, env_name):
    pickle.dump(clusterer, open("fracos_clusters/" + env_name + "/clusterers/" + "clusterer{}.p".format(cluster_level), "wb"))
    pickle.dump(clusters, open("fracos_clusters/" + env_name + "/clusters/" + "clusters{}.p".format(cluster_level), "wb"))
    pickle.dump(concat_fractures, open("fracos_clusters/" + env_name + "/other/" + "concat_fractures{}.p".format(cluster_level), "wb"))    
    pickle.dump(concat_trajs, open("fracos_clusters/" + env_name + "/other/" + "concat_trajs{}.p".format(cluster_level), "wb")) 
    if NN is not None:
        torch.save(NN.state_dict(), "fracos_clusters/"+ env_name + "/NNs/" + "NN_state_dict_{}.pth".format(cluster_level))
    pickle.dump(model_args, open("fracos_clusters/" + env_name + "/NN_args/" + "NN_args_{}.p".format(cluster_level), "wb"))

    if method is not None:
        pickle.dump(method.cypher, open("fracos_clusters/" + env_name + "/cluster_cyphers/" + "cypher_{}.p".format(cluster_level), "wb"))
        pickle.dump(method.reverse_cypher, open("fracos_clusters/" + env_name + "/cluster_reverse_cyphers/" + "cypher_{}.p".format(cluster_level), "wb"))

def refactor_trajectories(trajectories, fracos_agent, env_name, env, depth, gen_strength,
                          max_clusters_per_clusterer, device="cuda", chain_length=2, NN_predict=False):
    
    # This should use the exact clusters and not the NN predictions? 
    
    # stop compliling errors?
    
    refac_trajs = []
    traj_count = 0
    for traj in trajectories:
        print("starting {} of {} ".format(traj_count, len(trajectories)))
        traj_count += 1
        refac_traj = []
        option_found = False
        count = 0
        for move_idx in range(len(traj)-(chain_length-1)):
            move = traj[move_idx]
            if option_found and count < chain_length:
                # skip the next state and the next immediate analysis 
                option_found = False
                count += 1
            else:
                if "MetaGridEnv" in env_name:
                    state = move[0]
                    obs_0 = state[:49]
                    obs_0 = torch.tensor(obs_0).to(device)
                    obs_1 = state[49:]
                    obs_1 = torch.tensor(obs_1).to(device)
                    if fracos_agent.vae_path:
                        obs_0_enc = fracos_agent.encoder(obs_0.float().reshape(1,1,7,7))[0][0]
                    else:
                        obs_0_enc = obs_0
                        
                    obs_0_enc = obs_0_enc.detach().to("cpu")
                    obs_1 = obs_1.detach().to("cpu")
                    
                    state_c = np.concatenate((obs_0_enc, obs_1))
                    
                elif env_name == "ProcGen":
                    print("not implemented yet")
                else:
                    state_c = move[0]
                
                search_term = np.concatenate((state_c, traj[move_idx][1]))
                for ap in range(chain_length-1): # This just finds the chain
                    search_term = np.concatenate((search_term, traj[move_idx+ap+1][1])) # +ap+1 because the range will start from 0
                    
                # perform search here
                
                # search_term = torch.tensor(search_term)
                # search_term = search_term.float()
                # search_term = search_term.to(device)
            
                action_cypher=None
                for clstrr_idx in range(len(fracos_agent.clusterers)):
                    try:
                        if NN_predict:
                            with torch.no_grad():
                                predict_proba = fracos_agent.clstrr_NNs[clstrr_idx](search_term).squeeze()
                            # print("after preds = ", time.perf_counter()-NN_before)
                            cluster_label = torch.softmax(predict_proba, dim=0).argmax(dim=0)
                            # print("after softmax =", time.perf_counter()-NN_before)
                            cluster_label = cluster_label.cpu()
                            # print("after to cpu =", time.perf_counter()-NN_before)
                            strength = np.max(np.array(torch.softmax(predict_proba, dim=0).cpu()), axis=0)
                        
                        else:
                            clusterer = fracos_agent.clusterers[clstrr_idx]
                            cluster_labels, strengths = hdbscan.approximate_predict(clusterer, [search_term])
                            cluster_label = cluster_labels[0]
                            strength = strengths[0]
                        
                        if (cluster_label in fracos_agent.clusters[clstrr_idx]) and (strength >= (1 - gen_strength)):
                            # need to find where this particular option is and then decypher ... 
                            loc_label = fracos_agent.clusters[clstrr_idx].index(cluster_label)
                            action = env.action_space.n
                            for cl in range(clstrr_idx):
                                action += len(fracos_agent.clusters[cl])
                            action += loc_label
                            action_cypher = fracos_agent.cypher[action]
                    except:
                        pass # Depending on the refactoring from prim or from pre-loaded 
                        
                if action_cypher is not None:
                    refac_traj.append([state_c, action_cypher])
                    option_found = True
                    count = 0
                else:
                    # Need to add the old move as the new cypher action.
                    # old_fracos_agent = copy.deepcopy(fracos_agent)
                    fracos_agent.current_depth = fracos_agent.current_depth-1
                    fracos_agent.clusterers, fracos_agent.clusters, _, _, _ = fracos_agent.get_clusters(env_name)
                    fracos_agent.cypher, fracos_agent.reverse_cypher = fracos_agent.gen_cypher(fracos_agent.current_depth)
                    fracos_agent.get_clusters(env_name)
                    
                    action_revcypher = fracos_agent.reverse_cypher[tuple(move[1])]
                    
                    fracos_agent.current_depth = fracos_agent.current_depth+1
                    fracos_agent.clusterers, fracos_agent.clusters, _, _, _ = fracos_agent.get_clusters(env_name)
                    fracos_agent.cypher, fracos_agent.reverse_cypher = fracos_agent.gen_cypher(fracos_agent.current_depth)
                    fracos_agent.get_clusters(env_name)
                    
                    action_cypher = fracos_agent.cypher[action_revcypher]
                    refac_traj.append([state_c, action_cypher])
                        
                if (move_idx == len(traj)-2) and (option_found == False): # -2 because already on -1 and then want one less
                    old_fracos_agent = copy.deepcopy(fracos_agent)
                    old_fracos_agent.current_depth = fracos_agent.current_depth-1
                    old_fracos_agent.clusterers, old_fracos_agent.clusters, _, _, _ = old_fracos_agent.get_clusters(env_name)
                    old_fracos_agent.cypher, old_fracos_agent.reverse_cypher = old_fracos_agent.gen_cypher(old_fracos_agent.current_depth)
                    old_fracos_agent.get_clusters(env_name)
                    
                    action_revcypher = old_fracos_agent.reverse_cypher[tuple(traj[move_idx+1][1])]
                    action_cypher = fracos_agent.cypher[action_revcypher]
                    refac_traj.append([state_c, action_cypher])
                        
        refac_trajs.append(refac_traj)
    
    return refac_trajs

def offline_cluster_compress_pipeline(prim_trajectories, all_ep_rewards, failure_min, max_depth=4, chain_length=2,
                                      min_cluster_size=10, env_name="MetaGridEnv", vae_path=None,
                                      max_clusters_per_clusterer=50, gen_strength=0.33, a_pre_enc=False):
    
    # How can we deal with the action_one_hot_encodings?
    # Define them at the start? Can we keep adding to them, define parameters for clusterer?
    
    trajectories = prim_trajectories
    
    if "procgen" in env_name:
        env = gym.make(env_name)

    elif env_name == "MetaGridEnv_2026":
        env = gym.make("MetaGridEnv/metagrid-v0", domain_size=[14,14], seed=2026)
        plt.imshow(env.env_master.domain)
        plt.show()
    elif env_name == "MetaGridEnv":
        env = gym.make("MetaGridEnv/metagrid-v0", domain_size=[14,14])
        plt.imshow(env.env_master.domain)
        plt.show()
    elif env_name == "MetaGridEnv_Josh_grid":
        env = gym.make("MetaGridEnv/metagrid-v0", domain_size=[14,14], style="Josh_grid")
        plt.imshow(env.env_master.domain)
        plt.show()
    elif env_name == "MetaGridEnv_Four_rooms":
        env = gym.make("MetaGridEnv/metagrid-v0", domain_size=[14,14], style="Four_rooms")
        plt.imshow(env.env_master.domain)
        plt.show()
    else:
        env = gym.make(env_name)
    
    from method.fracos_QL import fracos_QL
    from method.fracos_PPO import fracos_PPO
    
    for depth in range(max_depth):
        
        # dimension
        action_dim = env.action_space.n
        try:
            initial_state = env.env_master.domain
        except:
            initial_state, _ = env.reset()
            
        state_dim = env.observation_space.shape[0]
        
    
        all_fractures, corre_traj = create_fractures(trajectories, env_name, chain_length=chain_length
                                                    ,vae_path=vae_path, a_pre_enc=a_pre_enc
                                                    )
        
        print("fractures completed")
        clusterer = create_clusterer(all_fractures, MIN_CLUSTER_SIZE=min_cluster_size)
        
        print("clusters created")
        
        concat_fractures = sum(all_fractures, [])
        concat_trajs = sum(corre_traj, [])
        
        all_s_f = get_all_s_f_index(all_fractures, all_ep_rewards, failure_std_threshold=None,
                                    use_std=False, failure_min=failure_min)
        
        print("success and failure determined")
        
        clusterer, top_cluster, all_success_clusters,\
                ordered_cluster_pi_dict, best_clusters_list = \
                        cluster_PI_compression(clusterer, concat_fractures, all_s_f, trajectories,
                                                chain_length=chain_length, max_cluster_returns=480, 
                                                min_PI_score = 0.1)

        print("PI compression finished")
        
        save_all_clusterings(clusterer, best_clusters_list, concat_fractures, concat_trajs,
                             None, None, cluster_level=depth, env_name=env_name)
        
        
        fracos_agent = fracos_PPO(state_dim, action_dim, 0.0002,
                                      0.0002, 0.99, 40, 0.2,
                                      env_name, max_depth, chain_length=chain_length,
                                      max_clusters_per_clusterer=max_clusters_per_clusterer,
                                      gen_strength=gen_strength, current_depth=depth+1,
                                      vae_path=vae_path)
    
        NN = create_NN_clusters(concat_fractures, clusterer.labels_, clusterer, env,
                                fracos_agent.all_possible_action_combs[depth+1], 
                                verbose=True, max_epochs=30000, gen_str=gen_strength)
        
        if -1 not in set(clusterer.labels_):
            new_clusterer_labels = set(clusterer.labels_).add(-1)
        else:
            new_clusterer_labels = set(clusterer.labels_)
        
        model_args = [len(concat_fractures[0]), len(new_clusterer_labels)]
        
        
        save_all_clusterings(clusterer, best_clusters_list, concat_fractures, concat_trajs,
                             NN, model_args, cluster_level=depth, env_name=env_name)
        
        ## create fracos_agent
        
        
        # fracos_agent = fracos_QL(state_dim, action_dim, 0.99, env_name, max_depth, initial_state,
        #                          vae_path=vae_path, current_depth=depth+1, gen_strength=gen_strength,
        #                          max_clusters_per_clusterer=max_clusters_per_clusterer,
        #                          chain_length=chain_length)
        
        fracos_agent = fracos_PPO(state_dim, action_dim, 0.0002,
                                      0.0002, 0.99, 40, 0.2,
                                      env_name, max_depth, chain_length=chain_length,
                                      max_clusters_per_clusterer=max_clusters_per_clusterer,
                                      gen_strength=gen_strength, current_depth=depth+1,
                                      vae_path=vae_path)
        
        
        trajectories = refactor_trajectories(trajectories, fracos_agent, env_name, env,
                              gen_strength=gen_strength, max_clusters_per_clusterer=max_clusters_per_clusterer,
                              depth=depth, chain_length=chain_length)
        
    pass

def offline_cluster_compress_prim_pipeline(prim_trajectories, all_ep_rewards, failure_min, max_depth=4, chain_length=2,
                                      min_cluster_size=10, env_name="MetaGridEnv", vae_path=None,
                                      max_clusters_per_clusterer=50, gen_strength=0.33, a_pre_enc=False,
                                      NN_predict=False):
    
    # This performs the same as the offline_cluster_compress_pipeline, but can 
    # perform all building from primitive only actions. Useful for trained agents
    # such as the trained Procgen or other.
    
    trajectories = prim_trajectories
    
    if "procgen" in env_name:
        env = gym.make(env_name)

    elif env_name == "MetaGridEnv_2026":
        env = gym.make("MetaGridEnv/metagrid-v0", domain_size=[14,14], seed=2026)
        plt.imshow(env.env_master.domain)
        plt.show()
    elif env_name == "MetaGridEnv":
        env = gym.make("MetaGridEnv/metagrid-v0", domain_size=[14,14])
        plt.imshow(env.env_master.domain)
        plt.show()
    elif env_name == "MetaGridEnv_Josh_grid":
        env = gym.make("MetaGridEnv/metagrid-v0", domain_size=[14,14], style="Josh_grid")
        plt.imshow(env.env_master.domain)
        plt.show()
    elif env_name == "MetaGridEnv_Four_rooms":
        env = gym.make("MetaGridEnv/metagrid-v0", domain_size=[14,14], style="Four_rooms")
        plt.imshow(env.env_master.domain)
        plt.show()
    else:
        env = gym.make(env_name)
    
    from method.fracos_QL import fracos_QL
    from method.fracos_PPO import fracos_PPO
    
    for depth in range(max_depth):
        
        # dimension
        action_dim = env.action_space.n
        try:
            initial_state = env.env_master.domain
        except:
            initial_state, _ = env.reset()
            
        try:
            state_dim = env.observation_space.shape[0]
        except:
            print("It is believed that the observation space is one dimensional -- determined by Discrete(xxx). If you think this is incorrect please adjust.")
            state_dim = 1
        
    
        all_fractures, corre_traj = create_fractures(trajectories, env_name, chain_length=chain_length
                                                    ,vae_path=vae_path, a_pre_enc=a_pre_enc
                                                    )
        
        print("fractures completed")
        clusterer = create_clusterer(all_fractures, MIN_CLUSTER_SIZE=min_cluster_size)
        
        print("clusters created")
        
        concat_fractures = sum(all_fractures, [])
        concat_trajs = sum(corre_traj, [])
        
        all_s_f = get_all_s_f_index(all_fractures, all_ep_rewards, failure_std_threshold=None,
                                    use_std=False, failure_min=failure_min)
        
        print("success and failure determined")
        
        clusterer, top_cluster, all_success_clusters,\
                ordered_cluster_pi_dict, best_clusters_list = \
                        cluster_PI_compression(clusterer, concat_fractures, all_s_f, trajectories,
                                                chain_length=chain_length, max_cluster_returns=480, 
                                                min_PI_score = 0.1)

        print("PI compression finished")
        
        save_all_clusterings(clusterer, best_clusters_list, concat_fractures, concat_trajs,
                             None, None, None, cluster_level=depth, env_name=env_name)
        
        
        
        # fracos_agent = fracos_QL(state_dim, action_dim, 0.99, env_name, depth, initial_state,
        #                           vae_path=vae_path, current_depth=depth+1, gen_strength=gen_strength,
        #                           max_clusters_per_clusterer=max_clusters_per_clusterer,
        #                           chain_length=chain_length)
        fracos_agent = fracos_PPO(state_dim, action_dim, initial_state, 0.0002,
                                      0.0002, 0.99, 40, 0.2,
                                      env_name, depth, chain_length=chain_length,
                                      max_clusters_per_clusterer=max_clusters_per_clusterer,
                                      gen_strength=gen_strength, current_depth=depth,
                                      vae_path=vae_path)
    
        NN, model_args = create_NN_clusters(concat_fractures, clusterer.labels_, clusterer, env,
                                fracos_agent.all_possible_action_combs[depth], 
                                verbose=True, max_epochs=50000, gen_str=gen_strength,
                                vae_path=vae_path)
        
        
        
        
        save_all_clusterings(clusterer, best_clusters_list, concat_fractures, concat_trajs,
                             NN, model_args, fracos_agent, cluster_level=depth, env_name=env_name)
        
        ## create fracos_agent
        
        
        # fracos_agent = fracos_QL(state_dim, action_dim, 0.99, env_name, depth, initial_state,
        #                           vae_path=vae_path, current_depth=depth+1, gen_strength=gen_strength,
        #                           max_clusters_per_clusterer=max_clusters_per_clusterer,
        #                           chain_length=chain_length)
        
        fracos_agent = fracos_PPO(state_dim, action_dim, initial_state, 0.0002,
                                      0.0002, 0.99, 40, 0.2,
                                      env_name, depth+1, chain_length=chain_length,
                                      max_clusters_per_clusterer=max_clusters_per_clusterer,
                                      gen_strength=gen_strength, current_depth=depth+1,
                                      vae_path=vae_path)
        
        print("Refactoring trajectories")
        
        trajectories = refactor_trajectories(trajectories, fracos_agent, env_name, env,
                              gen_strength=gen_strength, max_clusters_per_clusterer=max_clusters_per_clusterer,
                              depth=depth, chain_length=chain_length, NN_predict=NN_predict)
        
    pass

if __name__ == "__main__":
    
    try:
        register( id="MetaGridEnv/metagrid-v0",
                  entry_point="metagrid_gymnasium_wrapper:MetaGridEnv")

    except:
        print("MetaGridEnv already registered, skipping")
    
    
    # #### ONLY RUN THIS CELL IF YOU ARE LOADING TRAJECTORIES
    
    # env_name = "LunarLander-v2"
    
    env_name = "MetaGridEnv"
    
    # # CHANGE DEPENDING ON CLUSTER LEVEL!
    cluster_level = 0
    saved_traj_dir = "trajectories/opt/{}/{}".format(env_name,cluster_level)
    
    optimal_ep_rewards = []
    optimal_trajectories = []
    for file in os.listdir(saved_traj_dir + "/rew"): # rew and traj are the same
        if file == "bash.sh":
            pass
        else:
            rew_path = saved_traj_dir + "/rew/" + file
            traj_path = saved_traj_dir + "/traj/" + file
            rew_content = pickle.load(open(rew_path, "rb"))
            traj_content = pickle.load(open(traj_path, "rb"))
            try:
    #             print(traj_content)
    #             np.array(traj_content)
                optimal_ep_rewards += rew_content
                optimal_trajectories += traj_content
            except:
                print("there is an error in ", file)
    
    all_trajectories = optimal_trajectories
    all_ep_rewards = optimal_ep_rewards
    
    print("loaded trajectories")
    
    ######## unhash below if wanting to create fractures and NN fractures #######
    
    # print("creating fractures")
    # CHAIN_LENGTH=2
    # all_fractures, corre_traj = create_fractures(all_trajectories, chain_length=CHAIN_LENGTH
    #                                             #,vae_path=None
    #                                             )
    # concat_fractures = sum(all_fractures, [])
    # concat_trajs = sum(corre_traj, [])
    
    # MIN_CLUSTER_SIZE = 10 #int(len(all_trajectories)/30)
    
    
    # print("clustering")
    # clusterer = create_clusterer(all_fractures, MIN_CLUSTER_SIZE=MIN_CLUSTER_SIZE)
    
    # # first we find out which of our trajectories are failures
    # # Use the arguments to make this function do as you wish.
    # all_s_f = get_all_s_f_index(all_fractures, all_ep_rewards, failure_std_threshold=None,
    #                             use_std=False, failure_min=0.94)
    
    
    # from utils.compression import cluster_PI_compression
    # from method.fracos_PPO import fracos_PPO
    
    # print("PI")
    # # we already have our clusterer from above.
    # clusterer, top_cluster, all_success_clusters,\
    #         ordered_cluster_pi_dict, best_clusters_list = \
    #                 cluster_PI_compression(clusterer, concat_fractures, all_s_f, all_trajectories,
    #                                         chain_length=CHAIN_LENGTH, max_cluster_returns=480, 
    #                                         min_PI_score = 0.25)
    
    # clusters = best_clusters_list
    
    # # ####  visualize clusters ###############:
        
    # # ## ALL SETTING UP THE AGENT AND END ###
    # # MAX_DEPTH = 3
    # # CURRENT_DEPTH = 1
    # # CHAIN_LENGTH = 2
    
    # # try:
    # #     register( id="MetaGridEnv/metagrid-v0",
    # #       entry_point="metagrid_gymnasium_wrapper:MetaGridEnv")
    # # except:
    # #     print("MetaGridEnv is already registered")
    
    # # if env_name == "MetaGridEnv":
    # #     env = gym.make(env_name+"/metagrid-v0", domain_size=[14,14])
    # # else:
    # #     env = gym.make(env_name)
        
    # # state_dim = env.observation_space.shape[0]
    # # action_dim = env.action_space.n
    # # lr_actor = 0.0003       # learning rate for actor network
    # # lr_critic = 0.001       # learning rate for critic network
    # # K_epochs = 40               # update policy for K epochs
    # # eps_clip = 0.2              # clip parameter for PPO
    # # gamma = 0.99                # discount factor
    
    # # fracos_ppo_agent = fracos_PPO(state_dim, action_dim, lr_actor,
    # #                               lr_critic, gamma, K_epochs, eps_clip,
    # #                               env_name, MAX_DEPTH, current_depth=CURRENT_DEPTH, 
    # #                               trash_r = 0, max_clusters_per_clusterer=100, chain_length=CHAIN_LENGTH,
    # #                               vae_path="/home/x4nno/Documents/PhD/FRACOs_v4/pretrained/VAE/MetaGridEnv/vae_cnn.torch")
    
    # ### VISUALIZATION OF THE CLUSTERS ###
    
    # # for i in range(len(clusters)):
    # #     visualize_clusters(concat_fractures, concat_trajs, clusterer, clusters[i],
    # #                         fracos_ppo_agent, samples=5, MAX_DEPTH=MAX_DEPTH,
    # #                         chain_length=2)

    # ####### NN and testing ##############
    
    
    # # pickle.dump(clusterer, open("/home/x4nno/Documents/PhD/FRACOs_v4/fracos_clusters/" + env_name + "/clusterers/" + "clusterer{}.p".format(cluster_level), "wb"))
    # # pickle.dump(clusters, open("/home/x4nno/Documents/PhD/FRACOs_v4/fracos_clusters/" + env_name + "/clusters/" + "clusters{}.p".format(cluster_level), "wb"))
    # # pickle.dump(concat_fractures, open("/home/x4nno/Documents/PhD/FRACOs_v4/fracos_clusters/" + env_name + "/other/" + "concat_fractures{}.p".format(cluster_level), "wb"))    
    # # pickle.dump(concat_trajs, open("/home/x4nno/Documents/PhD/FRACOs_v4/fracos_clusters/" + env_name + "/other/" + "concat_trajs{}.p".format(cluster_level), "wb")) 
    
    # # concat_fractures = pickle.load(open("fracos_clusters/MetaGridEnv/other/concat_fractures{}.p".format(cluster_level), "rb"))
    # # clusterer = pickle.load(open("fracos_clusters/MetaGridEnv/clusterers/clusterer{}.p".format(cluster_level), "rb"))
    
    # X_train, X_test, y_train, y_test = train_test_split(concat_fractures, 
    #                                                     clusterer.labels_, 
    #                                                     stratify=clusterer.labels_,
    #                                                     random_state=1)
    
    # print("training NN")
    # NN = create_NN_clusters(concat_fractures, clusterer.labels_, verbose=True, max_epochs=5000)
    
    # # torch.save(NN.state_dict(), "/home/x4nno/Documents/PhD/FRACOs_v4/fracos_clusters/"+ env_name + "/NNs/" + "NN_state_dict_{}.pth".format(cluster_level))
    
    # model_args = [len(concat_fractures[0]), len(set(clusterer.labels_))]
    
    # # pickle.dump(model_args, open("/home/x4nno/Documents/PhD/FRACOs_v4/fracos_clusters/" + env_name + "/NN_args/" + "NN_args_{}.p".format(cluster_level), "wb"))
    
    ###### FINISH OF UNHASH FOR CREATING NN CLUSTERS ########
    
    ## (2) BELOW WILL CREATE CLUSTERS FROM THE OFFLINE PRIMITIVE TRAJECTORIES ONLY ## 
    
    offline_cluster_compress_prim_pipeline(all_trajectories, all_ep_rewards, failure_min=0.95,
                                      max_depth=3, max_clusters_per_clusterer=10,
                                      env_name=env_name, min_cluster_size=20,
                                      #vae_path="/home/x4nno/Documents/PhD/FRACOs_v4/pretrained/VAE/MetaGridEnv/vae_cnn.torch",
                                      vae_path=None,
                                      gen_strength=0.33,
                                      chain_length=2,
                                      NN_predict=False)
    
    ## FINISHED OF (2)
    
    ######### BELOW WILL TEST our visulaization of clusters in depth ########
    
    # # ## ALL SETTING UP THE AGENT AND END ###The only difference of why this would be worse is because the the clusters are not accurate enough? or there aren’t enough useful clusters? We should create more trajectories and more depth? We still see some benefit but it isn’t as pronounced as before.
    # from method.fracos_PPO import fracos_PPO
    # MAX_DEPTH = 2
    # CURRENT_DEPTH = 2
    # CHAIN_LENGTH = 2
    
    # try:
    #     register( id="MetaGridEnv/metagrid-v0",
    #       entry_point="metagrid_gymnasium_wrapper:MetaGridEnv")
    # except:
    #     print("MetaGridEnv is already registered")
    
    # if env_name == "MetaGridEnv":
    #     env = gym.make(env_name+"/metagrid-v0", domain_size=[14,14])
    # else:
    #     env = gym.make(env_name)
        
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.n
    # lr_actor = 0.0003       # learning rate for actor network
    # lr_critic = 0.001       # learning rate for critic network
    # K_epochs = 40               # update policy for K epochs
    # eps_clip = 0.2              # clip parameter for PPO
    # gamma = 0.99                # discount factor
    # max_clusters_per_clusterer=50
    # gen_strength=0.1 
        
    # fracos_agent = fracos_PPO(state_dim, action_dim, lr_actor,
    #                                   lr_critic, gamma, K_epochs, eps_clip,
    #                                   env_name, MAX_DEPTH, chain_length=CHAIN_LENGTH,
    #                                   max_clusters_per_clusterer=max_clusters_per_clusterer,
    #                                   gen_strength=gen_strength, current_depth=CURRENT_DEPTH,
    #                                   vae_path=None)
    
    
    # visualize_clusters_deep(CURRENT_DEPTH, 58
    #                ,
    #                fracos_agent, vae=False, samples=5, MAX_DEPTH=MAX_DEPTH,
    #               chain_length=2, from_offline_prim=True, env_name=env_name)
    