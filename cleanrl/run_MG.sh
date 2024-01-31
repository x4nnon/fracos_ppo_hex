#!/bin/bash

# hyperparam sweeps for our methods 


python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id LunarLander-v2 --current_depth=0 --wandb_project_name LL_seed_comparisons --total_timesteps=600000 --num_steps=512&
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 2 --env-id LunarLander-v2 --current_depth=0 --wandb_project_name LL_seed_comparisons --total_timesteps=600000 --num_steps=512&
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 3 --env-id LunarLander-v2 --current_depth=0 --wandb_project_name LL_seed_comparisons --total_timesteps=600000 --num_steps=512&
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 4 --env-id LunarLander-v2 --current_depth=0 --wandb_project_name LL_seed_comparisons --total_timesteps=600000 --num_steps=512&
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 5 --env-id LunarLander-v2 --current_depth=0 --wandb_project_name LL_seed_comparisons --total_timesteps=600000 


python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id LunarLander-v2 --current_depth=1 --wandb_project_name LL_seed_comparisons --total_timesteps=600000 --num_steps=512&
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 2 --env-id LunarLander-v2 --current_depth=1 --wandb_project_name LL_seed_comparisons --total_timesteps=600000 --num_steps=512&
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 3 --env-id LunarLander-v2 --current_depth=1 --wandb_project_name LL_seed_comparisons --total_timesteps=600000 --num_steps=512&
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 4 --env-id LunarLander-v2 --current_depth=1 --wandb_project_name LL_seed_comparisons --total_timesteps=600000 --num_steps=512&
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 5 --env-id LunarLander-v2 --current_depth=1 --wandb_project_name LL_seed_comparisons --total_timesteps=600000 


python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id LunarLander-v2 --current_depth=2 --wandb_project_name LL_seed_comparisons --total_timesteps=600000 --num_steps=512&
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 2 --env-id LunarLander-v2 --current_depth=2 --wandb_project_name LL_seed_comparisons --total_timesteps=600000 --num_steps=512&
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 3 --env-id LunarLander-v2 --current_depth=2 --wandb_project_name LL_seed_comparisons --total_timesteps=600000 --num_steps=512&
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 4 --env-id LunarLander-v2 --current_depth=2 --wandb_project_name LL_seed_comparisons --total_timesteps=600000 --num_steps=512&
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 5 --env-id LunarLander-v2 --current_depth=2 --wandb_project_name LL_seed_comparisons --total_timesteps=600000 
