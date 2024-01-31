#!/bin/bash

# hyperparam sweeps for our methods 

# 0
# seed 1
# lr
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --learning_rate 0.003
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --learning_rate 0.005
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --learning_rate 0.008
# ent_coef
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --ent_coef 0.05
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --ent_coef 0.1
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --ent_coef 0.2
# target_kl
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --target_kl 0.005
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --target_kl 0.01
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --target_kl 0.02
# envs
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --num_envs 8
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --num_envs 16
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --num_envs 32
# rollouts
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --num_steps 64
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --num_steps 128
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --num_steps 256

# seed 64
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --learning_rate 0.003
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --learning_rate 0.005
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --learning_rate 0.008
# ent_coef
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --ent_coef 0.05
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --ent_coef 0.1
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --ent_coef 0.2
# target_kl
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --target_kl 0.005
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --target_kl 0.01
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --target_kl 0.02
# envs
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --num_envs 8
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --num_envs 16
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --num_envs 32
# rollouts
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --num_steps 64
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --num_steps 128
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=0  --num_steps 256

# 1
# seed 1
# lr
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --learning_rate 0.003
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --learning_rate 0.005
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --learning_rate 0.008
# ent_coef
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --ent_coef 0.05
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --ent_coef 0.1
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --ent_coef 0.2
# target_kl
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --target_kl 0.005
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --target_kl 0.01
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --target_kl 0.02
# FraCOs_bias_factor
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --FraCOs_bias_factor 1
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --FraCOs_bias_factor 10
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --FraCOs_bias_factor 100
# envs
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --num_envs 8
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --num_envs 16
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --num_envs 32
# rollouts
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --num_steps 64
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --num_steps 128
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --num_steps 256

#seed 64
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --learning_rate 0.003
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --learning_rate 0.005
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --learning_rate 0.008
# ent_coef
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --ent_coef 0.05
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --ent_coef 0.1
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --ent_coef 0.2
# target_kl
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --target_kl 0.005
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --target_kl 0.01
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --target_kl 0.02
# FraCOs_bias_factor
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --FraCOs_bias_factor 1
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --FraCOs_bias_factor 10
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --FraCOs_bias_factor 100
# envs
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --num_envs 8
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --num_envs 16
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --num_envs 32
# rollouts
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --num_steps 64
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --num_steps 128
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=1  --num_steps 256

# 2
# seed 1
# lr
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --learning_rate 0.003
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --learning_rate 0.005
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --learning_rate 0.008
# ent_coef
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --ent_coef 0.05
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --ent_coef 0.1
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --ent_coef 0.2
# target_kl
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --target_kl 0.005
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --target_kl 0.01
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --target_kl 0.02
# FraCOs_bias_factor
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --FraCOs_bias_factor 1
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --FraCOs_bias_factor 10
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --FraCOs_bias_factor 100
# envs
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --num_envs 8
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --num_envs 16
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --num_envs 32
# rollouts
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --num_steps 64
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --num_steps 128
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --num_steps 256

# seed 64
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --learning_rate 0.003
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --learning_rate 0.005
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --learning_rate 0.008
# ent_coef
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --ent_coef 0.05
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --ent_coef 0.1
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --ent_coef 0.2
# target_kl
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --target_kl 0.005
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --target_kl 0.01
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --target_kl 0.02
# FraCOs_bias_factor
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --FraCOs_bias_factor 1
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --FraCOs_bias_factor 10
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --FraCOs_bias_factor 100
# envs
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --num_envs 8
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --num_envs 16
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --num_envs 32
# rollouts
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --num_steps 64
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --num_steps 128
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 64 --env-id MetaGridEnv/metagrid-v0 --current_depth=2  --num_steps 256