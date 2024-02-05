FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# easy way to use a specific version of python
RUN yes | conda create -n main python=3.9
# RUN yes | source /opt/conda/etc/profile.d/conda.sh
RUN yes | conda activate main

RUN yes | pip install -r cleanrl/requirements/requirements.txt 
RUN yes | pip uninstall gymnasium
RUN yes | pip install gymnasium/
RUN yes | pip install scikit-learn
RUN yes | pip install imblearn
RUN yes | conda install conda-forge::hdbscan
RUN yes | apt update
RUN yes | apt install wget gcc-8 unzip libssl1.0.0 software-properties-common
RUN yes | add-apt-repository ppa:ubuntu-toolchain-r/test
RUN yes | apt update
RUN apt-get install --only-upgrade libstdc++6
RUN pip install MetaGridEnv/
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install IPython
RUN pip install importlib-metadata
RUN pip install tyro
RUN pip install ufal.pybox2d

# Need to then do: 'wandb login'. and paste the api key.

# saved as tc2034/fracos_ppo
