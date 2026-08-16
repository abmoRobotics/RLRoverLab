ARG ISAAC_SIM_VERSION=6.0.1
ARG ISAAC_SIM_DIGEST=sha256:783444c706538aa76cf5126e911ddc5e618779e6105305ad4af4260362a30aa9
FROM nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_VERSION}@${ISAAC_SIM_DIGEST}

# Set default RUN shell to bash
SHELL ["/bin/bash", "-c"]

ARG ISAAC_SIM_PATH
ARG ISAAC_LAB_GIT_URL=https://github.com/isaac-sim/IsaacLab.git
ARG ISAAC_LAB_GIT_REF=v3.0.0-beta2.patch1
ARG ISAAC_LAB_GIT_SHA=ffff603eafc6b74264a5261cc0183d6a65390d78
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

# Install Isaac Lab from the verified Beta 2 Patch 1 release tag. Keep the
# expected commit separate so a moved or mistyped tag fails the image build.
RUN git clone ${ISAAC_LAB_GIT_URL} ${ISAAC_LAB_PATH} && \
    cd ${ISAAC_LAB_PATH} && \
    git checkout --detach ${ISAAC_LAB_GIT_REF} && \
    test "$(git rev-parse HEAD)" = "${ISAAC_LAB_GIT_SHA}" && \
    ln -sfn ${ISAAC_SIM_PATH} _isaac_sim && \
    ./isaaclab.sh --install 'rl[skrl],visualizer[kit]'

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
