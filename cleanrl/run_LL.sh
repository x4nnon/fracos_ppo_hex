#!/bin/bash

# hyperparam sweeps for our methods 


python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=0 --wandb_project_name LL_seed_comparisons --env_id  LunarLander-v2 --total_timesteps=600000 &
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 2 --env-id MetaGridEnv/metagrid-v0 --current_depth=0 --wandb_project_name LL_seed_comparisons --env_id  LunarLander-v2 --total_timesteps=600000 &
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 3 --env-id MetaGridEnv/metagrid-v0 --current_depth=0 --wandb_project_name LL_seed_comparisons --env_id  LunarLander-v2 --total_timesteps=600000 &
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 4 --env-id MetaGridEnv/metagrid-v0 --current_depth=0 --wandb_project_name LL_seed_comparisons --env_id  LunarLander-v2 --total_timesteps=600000 &
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 5 --env-id MetaGridEnv/metagrid-v0 --current_depth=0 --wandb_project_name LL_seed_comparisons --env_id  LunarLander-v2 --total_timesteps=600000 


python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=1 --wandb_project_name LL_seed_comparisons --env_id  LunarLander-v2 --total_timesteps=600000 &
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 2 --env-id MetaGridEnv/metagrid-v0 --current_depth=1 --wandb_project_name LL_seed_comparisons --env_id  LunarLander-v2 --total_timesteps=600000 &
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 3 --env-id MetaGridEnv/metagrid-v0 --current_depth=1 --wandb_project_name LL_seed_comparisons --env_id  LunarLander-v2 --total_timesteps=600000 &
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 4 --env-id MetaGridEnv/metagrid-v0 --current_depth=1 --wandb_project_name LL_seed_comparisons --env_id  LunarLander-v2 --total_timesteps=600000 &
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 5 --env-id MetaGridEnv/metagrid-v0 --current_depth=1 --wandb_project_name LL_seed_comparisons --env_id  LunarLander-v2 --total_timesteps=600000 


python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=2 --wandb_project_name LL_seed_comparisons --env_id  LunarLander-v2 --total_timesteps=600000 &
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 2 --env-id MetaGridEnv/metagrid-v0 --current_depth=2 --wandb_project_name LL_seed_comparisons --env_id  LunarLander-v2 --total_timesteps=600000 &
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 3 --env-id MetaGridEnv/metagrid-v0 --current_depth=2 --wandb_project_name LL_seed_comparisons --env_id  LunarLander-v2 --total_timesteps=600000 &
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 4 --env-id MetaGridEnv/metagrid-v0 --current_depth=2 --wandb_project_name LL_seed_comparisons --env_id  LunarLander-v2 --total_timesteps=600000 &
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 5 --env-id MetaGridEnv/metagrid-v0 --current_depth=2 --wandb_project_name LL_seed_comparisons --env_id  LunarLander-v2 --total_timesteps=600000 
