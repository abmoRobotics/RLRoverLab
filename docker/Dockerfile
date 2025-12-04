ARG ISAAC_SIM_VERSION=5.1.0
FROM nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_VERSION}

# Set default RUN shell to bash
SHELL ["/bin/bash", "-c"]

ARG ISAAC_SIM_PATH
ARG ISAAC_LAB_VERSION
# Set environment variables
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV ISAAC_LAB_PATH=/workspace/isaac_lab
ENV ROVER_LAB_PATH=/workspace/isaac_rover

# Install dependencies and remove cache (run as root, then switch back)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    libglib2.0-0 \
    ncurses-term \
    git-lfs \
    ffmpeg \
    sudo && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*

# Install this Isaac Lab and SKRL
RUN ${ISAAC_SIM_PATH}/python.sh -m pip install isaaclab==${ISAAC_LAB_VERSION} skrl gym torch torchvision numpy pymeshlab open3d gdown termcolor prettytable hidapi opencv-python wandb

# RUN rm -rf ${ISAAC_SIM_PATH}/exts/omni.isaac.core_archive/pip_prebundle/osqp* && \
#     ${ISAAC_SIM_PATH}/python.sh -m pip uninstall -y osqp && \
#     ${ISAAC_SIM_PATH}/python.sh -m pip install --no-cache-dir --no-binary :all: osqp


RUN git clone https://github.com/NVIDIA/Cosmos-Tokenizer.git \
    && cd Cosmos-Tokenizer \
    && git lfs pull \
    && ${ISAAC_SIM_PATH}/python.sh -m pip install .

RUN echo "alias isaaclab=${ISAAC_LAB_PATH}/isaaclab.sh" >> ${HOME}/.bashrc && \
    echo "alias python=${ISAAC_SIM_PATH}/python.sh" >> ${HOME}/.bashrc && \
    echo "alias python3=${ISAAC_SIM_PATH}/python.sh" >> ${HOME}/.bashrc && \
    echo "alias pip='${ISAAC_SIM_PATH}/python.sh -m pip'" >> ${HOME}/.bashrc && \
    echo "alias pip3='${ISAAC_SIM_PATH}/python.sh -m pip'" >> ${HOME}/.bashrc && \
    echo "alias tensorboard='${ISAAC_SIM_PATH}/python.sh ${ISAAC_SIM_PATH}/tensorboard'" >> ${HOME}/.bashrc && \
    echo "export TZ=$(date +%Z)" >> ${HOME}/.bashrc

# Configure /workspace/isaac_rover as a safe directory, so that Weights & Biases can be used.
RUN git config --global --add safe.directory /workspace/isaac_rover

# Set working directory
WORKDIR ${ROVER_LAB_PATH}
