# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
import sys
sys.path.append("MetaGridEnv/MetaGridEnv")
sys.path.append("/app")
sys.path.append("/app/fracos_ppo_hex")
sys.path.append("/opt/conda/envs/main/lib/python3.9/site-packages")

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import pickle
from utils.clustering import torch_classifier
import hdbscan 
import copy


import MetaGridEnv 
from gym.envs.registration import register 

register( id="MetaGridEnv/metagrid-v0",
          entry_point="metagrid_gymnasium_wrapper:MetaGridEnv")


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 3
    """seed of the experiment"""
    torch_deterministic: bool = False
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True #!!! change
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "LunarLander_fracos"
    """the wandb's project name"""
    wandb_entity: str = "tpcannon"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "LunarLander-v2"
    """the id of the environment"""
    total_timesteps: int = 400000
    """total timesteps of the experiments"""
    learning_rate: float = 5e-3
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 512
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 8
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.2
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.02
    """the target KL divergence threshold"""
    report_epoch: int = num_steps*num_envs
    """When to run a seperate epoch run to be reported. Make sure this is a multple of num_envs."""
    anneal_ent: bool = True
    """Toggle entropy coeff annealing"""


    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    
    # Below are fracos specific
    max_clusters_per_clusterer: int = 25
    """the maximum number of clusters at each hierarchy level"""
    current_depth: int = 2
    """this is the current level of hierarchy we are considering"""
    chain_length: int = 2
    """How long our option chains are"""
    NN_cluster_search: bool = True
    """Should we use NN to predict our clusters? if false will use hdbscan"""
    gen_strength: float = 0.05
    """This should be the strength of generalisation. for NN 0.1 seems good. for hdbscan 0.33"""    
    FraCOs_bias_factor: float = 10.0
    """How much to multiply the logit by to bias towards choosing the identified fracos"""

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def count_elements(nested_list):
    count = 0

    for element in nested_list:
        if isinstance(element, list):
            count += count_elements(element)
        else:
            count += 1

    return count

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    
class FraCOsAgent(nn.Module):
    def __init__(self, envs, env_id, max_clusters_per_clusterer,
                 current_depth, chain_length, NN_cluster_search, gen_strength,
                 discrete_search_cache={},
                 ):
        
        super().__init__()
        self.env_id = env_id
        self.gen_strength = gen_strength
        self.NN_cluster_search = NN_cluster_search
        self.action_prims = envs.single_action_space.n
        self.chain_length = chain_length
        self.discrete_search_cache = discrete_search_cache
        self.max_clusters_per_clusterer = max_clusters_per_clusterer
        self.current_depth = current_depth
        self.clusterers, self.clusters, self.clstrr_NNs, \
            self.cyphers, self.reverse_cyphers = self.get_clusters()
            
        self.all_possible_action_combs, self.apac_arr = self.get_all_possible_action_combs()
        
        self.total_action_dims = envs.single_action_space.n + count_elements(self.clusters)
        
         
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, self.total_action_dims), std=0.01),
        )
        
    def get_all_possible_action_combs(self):
        
        cluster_len_dict = {}
        cluster_len_dict[0] = self.action_prims
        for clusterer_idx in range(len(self.clusters)): # self.clusters is a list of clusters at each level
            cluster_len_dict[clusterer_idx+1] = len(self.clusters[clusterer_idx]) # because first idx is prim
        
        apac_arr = []
        all_possible_action_combs = []
        for idx_key in range(self.current_depth+1):
            cluster_possible_action_combs = []
            # create each search list
            
            cluster_action_len = 0
            for i in range(idx_key+1):
                cluster_action_len += cluster_len_dict[i]
            
            
            # it's been a long day, and my brain is too tired to find an elegant
            # solution to the mess below. Please feel free to fix it.
            
            # Need to loop over each level of clustering and add all possible to a list
            
            if self.chain_length == 2:
                for p_a1 in range(cluster_action_len):
                    for p_a2 in range(cluster_action_len):
                        p_ca1 = self.cyphers[idx_key][p_a1]
                        p_ca2 = self.cyphers[idx_key][p_a2]
                        cluster_possible_action_combs.append([p_ca1, p_ca2])
                        
            elif self.chain_length == 3:
                for p_a1 in range(cluster_action_len):
                    for p_a2 in range(cluster_action_len):
                        for p_a3 in range(cluster_action_len):
                            p_ca1 = self.cyphers[idx_key][p_a1]
                            p_ca2 = self.cyphers[idx_key][p_a2]
                            p_ca3 = self.cyphers[idx_key][p_a3]
                            cluster_possible_action_combs.append([p_ca1, p_ca2, p_ca3])
                            
                            
            elif self.chain_length == 4:
                for p_a1 in range(cluster_action_len):
                    for p_a2 in range(cluster_action_len):
                        for p_a3 in range(cluster_action_len):
                            for p_a4 in range(cluster_action_len):
                                p_ca1 = self.cyphers[idx_key][p_a1]
                                p_ca2 = self.cyphers[idx_key][p_a2]
                                p_ca3 = self.cyphers[idx_key][p_a3]
                                p_ca4 = self.cyphers[idx_key][p_a4]
                                cluster_possible_action_combs.append([p_ca1, p_ca2, p_ca3, p_ca4])
            
            elif self.chain_length == 5:
                for p_a1 in range(cluster_action_len):
                    for p_a2 in range(cluster_action_len):
                        for p_a3 in range(cluster_action_len):
                            for p_a4 in range(cluster_action_len):
                                for p_a5 in range(cluster_action_len):
                                    p_ca1 = self.cyphers[idx_key][p_a1]
                                    p_ca2 = self.cyphers[idx_key][p_a2]
                                    p_ca3 = self.cyphers[idx_key][p_a3]
                                    p_ca4 = self.cyphers[idx_key][p_a4]
                                    p_ca5 = self.cyphers[idx_key][p_a5]
                                    cluster_possible_action_combs.append([p_ca1, p_ca2, p_ca3, p_ca4, p_ca5])
            
            else:
                print("WE HAVEN'T IMPLEMENTED PAST chain_length=5 YET")
            
            all_possible_action_combs.append(cluster_possible_action_combs)
            
            first = True
            for inner_list in cluster_possible_action_combs:
                stacked_array = np.hstack(inner_list)
                if first:
                    stacked_arrays = stacked_array
                    first=False
                else:
                    stacked_arrays = np.vstack((stacked_arrays, stacked_array))
            apac_arr_temp = np.array(stacked_arrays)
            apac_arr.append(apac_arr_temp)
            
        return all_possible_action_combs, apac_arr
        
    def get_clusters(self):
        if "MetaGridEnv" in self.env_id:
            cluster_dir = "fracos_clusters/MetaGridEnv"
        else:
            cluster_dir = "fracos_clusters/{}".format(self.env_id)
            
            
        # check if exists and then make if not 
        if not os.path.exists(cluster_dir + "/clusters/"):
              os.makedirs(cluster_dir + "/clusters/")
              
        if not os.path.exists(cluster_dir + "/clusterers/"):
              os.makedirs(cluster_dir  + "/clusterers/")
        
        if not os.path.exists(cluster_dir  + "/other/"):
              os.makedirs(cluster_dir  + "/other/")
        
        if not os.path.exists(cluster_dir  + "/NNs/"):
              os.makedirs(cluster_dir  + "/NNs/")
              
        if not os.path.exists(cluster_dir  + "/NN_args/"):
              os.makedirs(cluster_dir  + "/NN_args/")
              
        if not os.path.exists(cluster_dir  + "/cluster_cyphers/"):
              os.makedirs(cluster_dir  + "/cluster_cyphers/")
              
        if not os.path.exists(cluster_dir  + "/cluster_reverse_cyphers/"):
              os.makedirs(cluster_dir  + "/cluster_reverse_cyphers/")
        
        
        # get all clustering saved
        clusters_path_list = os.listdir(cluster_dir  + "/clusters/")
        clusterers_path_list = os.listdir(cluster_dir  + "/clusterers/")
        clusterers_NN_path_list = os.listdir(cluster_dir  + "/NNs/")
        clusterers_NN_args_path_list = os.listdir(cluster_dir  + "/NN_args/")
        cluster_cyphers_path_list = os.listdir(cluster_dir  + "/cluster_cyphers/")
        cluster_reverse_cyphers_path_list = os.listdir(cluster_dir  + "/cluster_reverse_cyphers/")
        
        clusters_path_list.sort()
        clusterers_path_list.sort()
        clusterers_NN_path_list.sort()
        clusterers_NN_args_path_list.sort()
        cluster_cyphers_path_list.sort()
        cluster_reverse_cyphers_path_list.sort()
        
        # loop to get the clusters which are available.
        
        clusterers_list = [pickle.load(open(os.path.join(cluster_dir \
                            +  "/clusterers/", clusterer), "rb"))\
                               for clusterer in clusterers_path_list]
        
        clusters_list = [pickle.load(open(os.path.join(cluster_dir \
                             + "/clusters/", cluster), "rb"))\
                               for cluster in clusters_path_list]
        
        NNs_list = [torch.load(os.path.join(cluster_dir \
                             + "/NNs/", NN))\
                               for NN in clusterers_NN_path_list]
        
        NN_args_list = [pickle.load(open(os.path.join(cluster_dir \
                             + "/NN_args/", NN_args), "rb"))\
                               for NN_args in clusterers_NN_args_path_list]
            
        cypher_list = [pickle.load(open(os.path.join(cluster_dir \
                             + "/cluster_cyphers/", cyphers), "rb"))\
                               for cyphers in cluster_cyphers_path_list]
            
        reverse_cypher_list = [pickle.load(open(os.path.join(cluster_dir \
                             + "/cluster_reverse_cyphers/", r_cyphers), "rb"))\
                               for r_cyphers in cluster_reverse_cyphers_path_list]
        
        
        new_NN_list = []
        
        for i in range(len(NNs_list)):
            NN = torch_classifier(NN_args_list[i][0], NN_args_list[i][1])
            NN.load_state_dict(torch.load(os.path.join(cluster_dir \
                                +  "/NNs/", clusterers_NN_path_list[i])))
            NN.to(device)
            for param in NN.parameters():
                param.grad = None
            NN.eval()
            new_NN_list.append(NN)
                
        
        
        
        new_clusters_list = []
        for cl in clusters_list:
            if len(cl) > self.max_clusters_per_clusterer:
                cl = cl[:self.max_clusters_per_clusterer]
            new_clusters_list.append(cl)
        
        # return the clusterers and top clusters.
        return clusterers_list[:self.current_depth], \
            new_clusters_list[:self.current_depth], \
                new_NN_list[:self.current_depth], \
                    cypher_list[:self.current_depth+1], \
                        reverse_cypher_list[:self.current_depth+1] 
                        
    
    def initial_search(self, state):
        # We should be able to speed up the initial search -- it it very slow.

        # for debug / checking that the search makes sense.
        # compare with available clusters
        
        # plt.imshow(self.decoder(torch.tensor(state[:8]).to(device)).cpu().detach().numpy().reshape(7,7))
        # plt.show()
        # print(state[8:10]) 
        
        self.discrete_search_cache[state] = {}
        clstr_a_count = self.action_prims

        for clstrr_idx in range(len(self.clusterers)):
            # clustrr_idx will now correspond to those in the 
            # all possible action combs, and we want to search through all.
            search_terms = []

            apac_arr = self.apac_arr[clstrr_idx]
            
            search_state = state
            search_state = np.array(search_state)
            
            # we should be able to speed this up?
            search_state_tile = np.tile(search_state, 
                                        (len(self.all_possible_action_combs[clstrr_idx]),1))
            
            search_terms = np.hstack((search_state_tile, apac_arr))

            # Search within our indicative clusterer
            if self.NN_cluster_search:
                search_terms = torch.tensor(search_terms)
                search_terms = search_terms.float()
                search_terms = search_terms.to(device)
                with torch.inference_mode():
                    predict_proba = self.clstrr_NNs[clstrr_idx](search_terms).squeeze()
                # print("after preds = ", time.perf_counter()-NN_before)
                cluster_labels = torch.softmax(predict_proba, dim=1).argmax(dim=1)
                # print("after softmax =", time.perf_counter()-NN_before)
                cluster_labels = cluster_labels.cpu()
                # print("after to cpu =", time.perf_counter()-NN_before)
                strengths = np.max(np.array(torch.softmax(predict_proba.cpu().detach(), dim=1).cpu()), axis=1)

            else:
                cluster_labels, strengths = hdbscan.approximate_predict(self.clusterers[clstrr_idx], search_terms)

            cluster_labels = np.array(cluster_labels)
            
            # update to the discrete search?
            # need to use the cypher at the current depth for this
            
            
            for clstr_idx in range(len(self.clusters[clstrr_idx])):
                cypher_action = self.cyphers[self.current_depth][clstr_a_count]
                cypher_action = tuple(cypher_action)
                clstr_a_count += 1
                
                action = self.reverse_cyphers[self.current_depth][cypher_action]
                self.discrete_search_cache[state][action] = [None, None]
                top_strength = 0
                best_action_pair_match = [None, None]
                if (self.clusters[clstrr_idx][clstr_idx] in cluster_labels):
                        
                    indexes = np.where(cluster_labels == self.clusters[clstrr_idx][clstr_idx])
                    
                    
                    for idx in indexes[0]:
                        c_strength = strengths[idx]
                        action_pair = self.all_possible_action_combs[clstrr_idx][idx]
                        if (c_strength > top_strength) & (c_strength >= 1-self.gen_strength):
                            top_strength = c_strength
                            best_action_pair_match = action_pair
                        
                else:
                    pass
                            
                final_best_action_pair_match = []
                if best_action_pair_match[0] is not None:
                    for ba in best_action_pair_match:
                        # Turn back to a number
                        ba_n = self.reverse_cyphers[clstrr_idx][tuple(ba)]
                        # Turn into the correct cypher 
                        ba_c = self.cyphers[self.current_depth][ba_n]
                        final_best_action_pair_match.append(ba_c)
                else: final_best_action_pair_match = best_action_pair_match
                
                self.discrete_search_cache[state][action] = final_best_action_pair_match
                    
                    
                    
                        
            

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        
        # we have vectorized environments so we need to extract the information here?
        x_idx = 0
        logits = self.actor(x)
        for x_ind in x:
            
            # conduct our initial search
            x_tuple = tuple(x_ind.cpu().numpy())
            if x_tuple in self.discrete_search_cache.keys():
                pass
            else:
                self.initial_search(x_tuple)
            
            # find available_actions
            available_actions = []
            for pot_a in self.discrete_search_cache[x_tuple].keys():
                if isinstance(self.discrete_search_cache[x_tuple][pot_a][0], np.ndarray):
                    available_actions.append(pot_a)
            
            available_actions = list(available_actions)
            for i in range(self.action_prims): 
                available_actions.append(i)
            
            
            # we should change the logits to only have those which are available?
            for i in range(len(logits[x_idx])):
                if i not in available_actions:
                    logits[x_idx][i] = -1e6
                elif i > self.action_prims:
                    if logits[x_idx][i] < 0:
                        logits[x_idx][i] = logits[x_idx][i]/args.FraCOs_bias_factor
                    else:
                        logits[x_idx][i] = logits[x_idx][i]*args.FraCOs_bias_factor # logits can be negative too ..
            
            x_idx += 1
            
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    random.seed(args.seed)
    
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{args.current_depth}__{args.FraCOs_bias_factor}__{datetime.now()}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = FraCOsAgent(envs, args.env_id,
                        args.max_clusters_per_clusterer, 
                        args.current_depth, args.chain_length,
                        args.NN_cluster_search, args.gen_strength).to(device)
    
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    # old style storage
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    # obs = []
    # actions = []
    # logprobs = []
    # rewards = []
    # dones = []
    # values = []

    # TRY NOT TO MODIFY: start the game
    global_decisions = 0 
    global_step_truth = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    epoch = 0
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        if args.anneal_ent:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            ent_coef_now = frac * args.ent_coef
        else:
            ent_coef_now = args.ent_coef
            

        for step in range(0, args.num_steps):
            
            obs[step] = next_obs
            # obs.append(next_obs)
            dones[step] = next_done
            # dones.append(next_done)

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            # actions.append(action)
            logprobs[step] = logprob
            # logprobs.append(logprob)

            # TRY NOT TO MODIFY: execute the game and log data. !!! unfortunatly I need to modify
            # new style
            
            next_obs, reward, terminations, truncations, infos, total_steps_taken = envs.fracos_step_async(action.cpu().numpy(), next_obs, agent)
            global_step_truth += sum(total_steps_taken)
            
            # new style end 
            
            
            # old style

            # next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            
            # total_steps_taken = [step]
            # global_step_truth += args.num_envs
            
            # end old_style
            
            
            global_decisions += args.num_envs 
            
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step_truth}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step_truth)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step_truth)
            
            # where we get our epoch results from.
            if global_decisions % args.report_epoch == 0:
                with torch.no_grad():
                    epoch_len = args.report_epoch
                    test_envs = copy.deepcopy(envs)
                    test_envs.train = False
                    # test_seed = random.randint(1001,2000) #makes sure that the random int for train is below 1000
                    test_next_obs, _ = test_envs.reset() # random seedings
                    test_next_obs = torch.Tensor(test_next_obs).to(device)
                    all_test_rewards = 0
                    for ts in range(int(epoch_len/args.num_envs)):
                        test_action, test_logprob, _, test_value = agent.get_action_and_value(test_next_obs)
                        test_next_obs, test_reward, test_terminations, test_truncations, test_infos, total_steps_taken = envs.fracos_step_async(test_action.cpu().numpy(), test_next_obs, agent)
                        test_next_obs = torch.Tensor(test_next_obs).to(device)
                        all_test_rewards += sum(test_reward)
                    
                    average_all_test_rewards = all_test_rewards/epoch_len
                    writer.add_scalar("charts/epoch_returns", all_test_rewards, global_step_truth)
                    writer.add_scalar("charts/average_return_per_epoch_step", average_all_test_rewards, global_step_truth)
                    epoch += 1
                    

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                
                # this is to change the rewards from multiple negative for taking an option 
                # only changes for the update step and not for the reporting.
                # !!! MUST CHANGE THIS DEPENDANT ON ENVIRONEMENT !!!
                adjusted_rewards = rewards[t]
                # !!! for MetaGrid unhash this. For LL hash -- ideally we could do this for LL but unsure of complex rewards
                for i in range(len(adjusted_rewards)):
                    if -0.001 > adjusted_rewards[i] > -0.1:
                        adjusted_rewards[i] = -0.01
                
                delta = adjusted_rewards + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef_now * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_decisions)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_decisions)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_decisions)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_decisions)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_decisions)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_decisions)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_decisions)
        writer.add_scalar("losses/explained_variance", explained_var, global_decisions)
        print("SPS:", int(global_step_truth / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step_truth / (time.time() - start_time)), global_step_truth)
        writer.add_scalar("charts/decisions", global_decisions, global_step_truth)

        if global_step_truth >= args.total_timesteps:
            break
        
    envs.close()
    writer.close()
