FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

#update python to 3.9
RUN apt install build-essential zlib1g-dev \
RUN apt install wget
RUN wget https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tar.xz
RUN tar -xf Python-3.9.0.tar.xz
RUN cd Python-3.9.0
RUN .configure/
RUN make altinstall


RUN pip install -r cleanrl/requirements/requirements.txt 
RUN pip install scikit-learn
RUN pip install imblearn
RUN conda install conda-forge::hdbscan
RUN apt update
RUN apt install wget gcc-8 unzip libssl1.0.0 software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt update
RUN apt-get install --only-upgrade libstdc++6
RUN pip install MetaGridEnv/

# Need to then do: 'wandb login'. and paste the api key.
