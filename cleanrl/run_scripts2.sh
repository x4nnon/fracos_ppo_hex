#!/bin/bash

# hyperparam sweeps for our methods 


python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=0 --ent_coef 0.2 --gen_strength 0.33
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=1 --ent_coef 0.2 --gen_strength 0.33
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 1 --env-id MetaGridEnv/metagrid-v0 --current_depth=2 --ent_coef 0.2 --gen_strength 0.33

python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 2 --env-id MetaGridEnv/metagrid-v0 --current_depth=0 --ent_coef 0.2 --gen_strength 0.33
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 2 --env-id MetaGridEnv/metagrid-v0 --current_depth=1 --ent_coef 0.2 --gen_strength 0.33
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 2 --env-id MetaGridEnv/metagrid-v0 --current_depth=2 --ent_coef 0.2 --gen_strength 0.33

python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 3 --env-id MetaGridEnv/metagrid-v0 --current_depth=0 --ent_coef 0.2 --gen_strength 0.33
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 3 --env-id MetaGridEnv/metagrid-v0 --current_depth=1 --ent_coef 0.2 --gen_strength 0.33
python3 $(pwd)/cleanrl/cleanrl/ppo.py --seed 3 --env-id MetaGridEnv/metagrid-v0 --current_depth=2 --ent_coef 0.2 --gen_strength 0.33
