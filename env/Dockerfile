#####################################
# Default build type is defined here
#####################################
ARG TARGET="cpu"

##############################
# Base image for 'cpu' builds
##############################
FROM ubuntu:22.04 as cpu

# Default to Zsh for CPU builds
ARG USER_SHELL="/usr/bin/zsh"

# PyTorch index for CPU-only wheels
ENV TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"

################################################
# Base image for 'gpu9' builds, using CUDA v9.0
################################################
FROM nvcr.io/nvidia/cuda:9.0-devel-ubuntu16.04 as gpu9

# Default to Bash for GPU builds
ARG USER_SHELL="/bin/bash"

# PyTorch index for CUDA 11.8 wheels
ENV TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"

###########################################################
# Base image for 'gpu' builds, using a recent CUDA version
#----------------------------------------------------------
# Note that Nvidia no longer supports a 'latest' tag,
# so this needs to be maintained
###########################################################
FROM nvcr.io/nvidia/cuda:12.3.1-devel-ubuntu22.04 as gpu

# Default to Bash for GPU builds
ARG USER_SHELL="/bin/bash"

# PyTorch index for CUDA 12.1 wheels
ENV TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"

#####################################################
# Everything that follows is the same for all builds
#####################################################
FROM ${TARGET}
ARG TARGET
ENV DOCKER_TARGET=${TARGET}

################
# Set user info
################
ARG USERNAME
ARG USER_ID
ARG USER_GID
ARG USER_SHELL_DEFAULT=${USER_SHELL}

##################################################
# System update and system-level library installs
##################################################
RUN apt-get update -y && \
    apt-get upgrade -y && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends\
       ca-certificates  \
       build-essential \
       apt-utils \
       cmake \
       git \
       less \
       zsh \
       curl \
       vim \
       gfortran \
       zlib1g-dev \
       automake \
       autoconf \
       git \
       libtool \
       subversion \
       libatlas3-base \
       libtinfo6 \ 
       wget \
       unzip \
       locales && \
    rm -rf /var/lib/apt/lists/*

###############################################################
# Set some environment variables defining the user environment
###############################################################
ENV HOME=/u/home \
    TMP_PATH=${HOME}/dump \
    LANG=en_US.utf8

###########################################
# Set up locale (especially needed by Zsh)
###########################################
RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen && \
    locale-gen --purge --lang en_US && \
    locale-gen

#####################################################################################
# Install uv (fast Python package manager)
#------------------------------------------------------------------------------------
# uv replaces conda for managing the Python installation, virtual environment,
# and package installs.  It is much faster than conda/pip and avoids the
# Anaconda Terms-of-Service issue that blocks non-interactive Docker builds.
#####################################################################################
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

###########################################################################
# Add a user which will be mapped to the runtime user who builds the image
#--------------------------------------------------------------------------
# n.b.: It is assumed that the user building a container is the same as the
#       one who builds the image.
###########################################################################
RUN mkdir -p ${HOME} && \
    groupadd --force --gid ${USER_GID} ${USERNAME} && \
    useradd \
    --uid ${USER_ID} \
    --gid ${USER_GID} \
    --home-dir ${HOME} \
    --shell ${USER_SHELL_DEFAULT} \
    ${USERNAME} && \
    chown -R ${USER_ID}:${USER_GID} ${HOME}

#############################################################
# Switch to the user that we just created.  In what follows,
# we set-up their runtime environment.
#############################################################
USER ${USERNAME}
WORKDIR ${HOME}

#####################################################################################
# Install Python 3.11 via uv and create a virtual environment
#####################################################################################
ENV VIRTUAL_ENV=${HOME}/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
RUN uv python install 3.11
RUN uv venv ${VIRTUAL_ENV} --python 3.11

#####################################################################################
# Install PyTorch
#------------------------------------------------------------------------------------
# The TORCH_INDEX_URL is set per build target (cpu / gpu / gpu9) in the
# stage definitions above.
#####################################################################################
RUN uv pip install torch torchvision --index-url ${TORCH_INDEX_URL}

#################
# Install Poetry
#################
RUN uv pip install "poetry>=2.0,<3.0"
RUN poetry config virtualenvs.create false

################################################################
# Copy and install Poetry dependencies (but not the actual
# application, which will get installed by the entry_point
# script when we start the container)
################################################################
COPY pyproject.toml poetry.lock .
RUN poetry install --no-root --all-extras && \
    rm pyproject.toml poetry.lock

###############################################
# Activate the virtual environment in shells
###############################################
RUN echo "source ${VIRTUAL_ENV}/bin/activate" >> .bashrc
RUN echo "source ${VIRTUAL_ENV}/bin/activate" >> .zshrc

##############
# Set up bash
##############
RUN export PROMPT_COLOUR_PATH="\[\033[34m\]" && \
    export PROMPT_COLOUR_PROMPT="\[\033[35m\]" && \
    export PROMPT_COLOUR_RESET="\[\033[0m\]" && \
    echo PS1=\"${PROMPT_COLOUR_PATH}\\n\\w\\n${PROMPT_COLOUR_PROMPT}\\u@docker-${DOCKER_TARGET} $ ${PROMPT_COLOUR_RESET}\" >> .bashrc

#############
# Set up Zsh
#############
COPY env/zshrc ${HOME}/.zshrc_user
RUN echo source ${HOME}/.zshrc_user >> .zshrc
RUN git clone https://github.com/ohmyzsh/ohmyzsh.git ${HOME}/.oh-my-zsh && \
    sed -i 's/❯/%n@docker-${DOCKER_TARGET} ❯/g' ${HOME}/.oh-my-zsh/themes/refined.zsh-theme
ENV ZSH=${HOME}/.oh-my-zsh \
    PATH_TO_COPY=${PATH}

##########################
# Set-up the entry point
##########################
USER root
COPY env/entry_script.sh /
RUN chmod a+rx /entry_script.sh
# Make sure that the 'SuperNNova' path used here matches what gets used
# in the 'launch_docker.py' script.
# The venv is already on PATH, so poetry runs directly.
RUN echo "cd SuperNNova && poetry install --all-extras --only-root" >> /entry_script.sh
RUN echo su -m $USERNAME >> /entry_script.sh
ENTRYPOINT ["sh", "-c", "/entry_script.sh"]
