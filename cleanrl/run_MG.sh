#!/bin/bash

# hyperparam sweeps for our methods 


python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=0 --wandb_project_name MG_seed_comparisons --total_timesteps=200000 --num_steps=128 --NN_cluster_search=True --gen_strength=0.05 
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 2 --env-id MetaGridEnv/metagrid-v0 --current_depth=0 --wandb_project_name MG_seed_comparisons --total_timesteps=200000 --num_steps=128 --NN_cluster_search=True --gen_strength=0.05 
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 3 --env-id MetaGridEnv/metagrid-v0 --current_depth=0 --wandb_project_name MG_seed_comparisons --total_timesteps=200000 --num_steps=128  --NN_cluster_search=True --gen_strength=0.05 
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 4 --env-id MetaGridEnv/metagrid-v0 --current_depth=0 --wandb_project_name MG_seed_comparisons --total_timesteps=200000 --num_steps=128  --NN_cluster_search=True --gen_strength=0.05 
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 5 --env-id MetaGridEnv/metagrid-v0 --current_depth=0 --wandb_project_name MG_seed_comparisons --total_timesteps=200000 --num_steps=128 --NN_cluster_search=True


python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=1 --wandb_project_name MG_seed_comparisons --total_timesteps=200000 --num_steps=128  --NN_cluster_search=True --gen_strength=0.05
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 2 --env-id MetaGridEnv/metagrid-v0 --current_depth=1 --wandb_project_name MG_seed_comparisons --total_timesteps=200000 --num_steps=128 --NN_cluster_search=True --gen_strength=0.05
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 3 --env-id MetaGridEnv/metagrid-v0 --current_depth=1 --wandb_project_name MG_seed_comparisons --total_timesteps=200000 --num_steps=128 --NN_cluster_search=True --gen_strength=0.05
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 4 --env-id MetaGridEnv/metagrid-v0 --current_depth=1 --wandb_project_name MG_seed_comparisons --total_timesteps=200000 --num_steps=128 --NN_cluster_search=True --gen_strength=0.05
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 5 --env-id MetaGridEnv/metagrid-v0 --current_depth=1 --wandb_project_name MG_seed_comparisons --total_timesteps=200000 --num_steps=128 --NN_cluster_search=True --gen_strength=0.05


python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=2 --wandb_project_name MG_seed_comparisons --total_timesteps=200000 --num_steps=128 --NN_cluster_search=True --gen_strength=0.05 
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 2 --env-id MetaGridEnv/metagrid-v0 --current_depth=2 --wandb_project_name MG_seed_comparisons --total_timesteps=200000 --num_steps=128 --NN_cluster_search=True --gen_strength=0.05 
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 3 --env-id MetaGridEnv/metagrid-v0 --current_depth=2 --wandb_project_name MG_seed_comparisons --total_timesteps=200000 --num_steps=128 --NN_cluster_search=True --gen_strength=0.05 
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 4 --env-id MetaGridEnv/metagrid-v0 --current_depth=2 --wandb_project_name MG_seed_comparisons --total_timesteps=200000 --num_steps=128 --NN_cluster_search=True --gen_strength=0.05 
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 5 --env-id MetaGridEnv/metagrid-v0 --current_depth=2 --wandb_project_name MG_seed_comparisons --total_timesteps=200000 --num_steps=128 --NN_cluster_search=True --gen_strength=0.05
