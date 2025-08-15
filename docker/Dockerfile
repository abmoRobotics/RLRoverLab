ARG ISAAC_SIM_VERSION
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

# Install dependencies and remove cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    libglib2.0-0 \
    ncurses-term && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*



# Install this Isaac Lab and SKRL
RUN ${ISAAC_SIM_PATH}/python.sh -m pip install isaaclab==${ISAAC_LAB_VERSION} skrl==1.4.1 gym

# Clone and install Cosmos Tokenizer
RUN apt-get update && apt-get install -y git-lfs ffmpeg 

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
    echo "export TZ=$(date +%Z)" >> ${HOME}/.bashrc \
    echo "export TZ=$(date +%Z)" >> ${HOME}/.bashrc


# Configure /workspace/isaac_rover as a safe directory, so that Weights & Biases can be used.
RUN git config --global --add safe.directory /workspace/isaac_rover

# Set working directory
WORKDIR ${ROVER_LAB_PATH}
