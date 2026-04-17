ARG ISAAC_SIM_VERSION=6.0.0-dev2
FROM nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_VERSION}

# Set default RUN shell to bash
SHELL ["/bin/bash", "-c"]

ARG ISAAC_SIM_PATH
ARG ISAAC_LAB_GIT_URL=https://github.com/isaac-sim/IsaacLab.git
ARG ISAAC_LAB_GIT_REF=a4a7602
# Set environment variables
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV ISAAC_LAB_PATH=/workspace/isaac_lab
ENV ROVER_LAB_PATH=/workspace/rlroverlab

# Switch to root for package installation
USER root

# Install dependencies and remove cache
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

# Install Isaac Lab from source.
# The ref can be either a tag (e.g. v3.0.0-beta) or a commit SHA (e.g. a4a7602).
RUN git clone ${ISAAC_LAB_GIT_URL} ${ISAAC_LAB_PATH} && \
    cd ${ISAAC_LAB_PATH} && \
    git checkout ${ISAAC_LAB_GIT_REF} && \
    ln -sfn ${ISAAC_SIM_PATH} _isaac_sim && \
    ./isaaclab.sh --install assets,contrib,mimic,newton,ov,physx,rl[rl_games,rsl_rl,sb3,skrl,robomimic],tasks,teleop,visualizers

# Install dependencies from setup.py (without installing the package itself)
# Copy as root then change ownership to isaac-sim user
COPY --chown=isaac-sim:isaac-sim setup.py pyproject.toml /tmp/rover_deps/
RUN mkdir -p /tmp/rover_deps/rover_envs && touch /tmp/rover_deps/rover_envs/__init__.py && \
    cd /tmp/rover_deps && \
    SKIP_ISAAC_SIM_INSTALL=true ${ISAAC_SIM_PATH}/python.sh -m pip install --user --no-cache-dir . && \
    rm -rf /tmp/rover_deps

# RUN rm -rf ${ISAAC_SIM_PATH}/exts/omni.isaac.core_archive/pip_prebundle/osqp* && \\
#     ${ISAAC_SIM_PATH}/python.sh -m pip uninstall -y osqp && \
#     ${ISAAC_SIM_PATH}/python.sh -m pip install --no-cache-dir --no-binary :all: osqp


RUN echo "alias isaaclab=${ISAAC_LAB_PATH}/isaaclab.sh" >> ${HOME}/.bashrc && \
    echo "alias python=${ISAAC_SIM_PATH}/python.sh" >> ${HOME}/.bashrc && \
    echo "alias python3=${ISAAC_SIM_PATH}/python.sh" >> ${HOME}/.bashrc && \
    echo "alias pip='${ISAAC_SIM_PATH}/python.sh -m pip'" >> ${HOME}/.bashrc && \
    echo "alias pip3='${ISAAC_SIM_PATH}/python.sh -m pip'" >> ${HOME}/.bashrc && \
    echo "alias tensorboard='${ISAAC_SIM_PATH}/python.sh ${ISAAC_SIM_PATH}/tensorboard'" >> ${HOME}/.bashrc && \
    echo "export TZ=$(date +%Z)" >> ${HOME}/.bashrc

# Configure /workspace/rlroverlab as a safe directory, so that Weights & Biases can be used.
RUN git config --global --add safe.directory /workspace/rlroverlab

# Set working directory
WORKDIR ${ROVER_LAB_PATH}
