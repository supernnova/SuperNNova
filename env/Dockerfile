#####################################
# Default build type is defined here
#####################################
ARG TARGET="cpu"

##############################
# Base image for 'cpu' builds
##############################
FROM ubuntu:22.04 as cpu

################################################
# Base image for 'gpu9' builds, using CUDA v9.0
################################################
FROM nvcr.io/nvidia/cuda:9.0-devel-ubuntu16.04 as gpu9

###########################################################
# Base image for 'gpu' builds, using a recent CUDA version
#----------------------------------------------------------
# Note that Nvidia no longer supports a 'latest' tag,
# so this needs to be maintained
###########################################################
FROM nvcr.io/nvidia/cuda:12.3.1-devel-ubuntu22.04 as gpu

#####################################################
# Everything that follows is the same for all builds
#####################################################
FROM ${TARGET}

################
# Set user info
################
ARG USERNAME
ARG USER_ID
ARG USER_GID
ARG USER_SHELL="/bin/bash"

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

###########################################################################
# Add a user which will be mapped to the runtime user who builds the image
#--------------------------------------------------------------------------
# n.b.: It is assumed that the user building a container is the same as the
#       one who builds the image.
###########################################################################
RUN mkdir -p ${HOME} && \
    groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd \
    --uid ${USER_ID} \
    --gid ${USER_GID} \
    --home-dir ${HOME} \
    --shell ${USER_SHELL} \
    ${USERNAME} && \
    chown -R ${USER_ID}:${USER_GID} ${HOME}

#############################################################
# Switch to the user that we just created.  In what follows, 
# we set-up their runtime environment.
#############################################################
USER ${USERNAME}
WORKDIR ${HOME}

#####################################################################################
# Instal Anaconda
#------------------------------------------------------------------------------------
# n.b.: we don't add ${CONDA_HOME}/lib and ${CONDA_HOME}/bin to 
# their respective paths here to avoid the following error:
# https://stackoverflow.com/questions/72103046/libtinfo-so-6-no-version-information-available-message-using-conda-environment
# If you do feel the need to do this, you may need to do the 
# following to deal with this:
# RUN ${CONDA} install -n ${CONDA_ENV_NAME} -c conda-forge --force-reinstall ncurses
#####################################################################################
ENV CONDA_HOME=${HOME}/.miniconda3
ENV CONDA=${CONDA_HOME}/bin/conda \
    CONDA_ENV_NAME=supernnova
RUN curl -o ${HOME}/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod a+x ${HOME}/miniconda.sh
RUN ${HOME}/miniconda.sh -b -p ${CONDA_HOME} && \
    rm ${HOME}/miniconda.sh

############################################################
# Create Conda environment
#-----------------------------------------------------------
# The use of a Conda env here simplifies the support of old OS
# & CUDA versions while using a modern Python version.  In the
# future, we may be able to drop support for old platforms;
# install Python with `apt-get`; and drop this conda env
############################################################
COPY env/conda_env.yml ${HOME}
RUN ${CONDA} config --set auto_activate_base false
# RUN mkdir -p ${HOME}/.conda/envs 
RUN ${CONDA} env create -f conda_env.yml 
RUN ${CONDA} clean -ya  
RUN rm conda_env.yml 

#################
# Install Poetry
#################
RUN ${CONDA} run -n ${CONDA_ENV_NAME} pip install poetry

########################################################################
# Copy and install Poetry dependencies (but not the actual application)
########################################################################
COPY pyproject.toml poetry.lock .
RUN ${CONDA} run -n ${CONDA_ENV_NAME} poetry install --no-root --all-extras && \
    rm pyproject.toml poetry.lock
RUN echo export PATH=${CONDA_HOME}/bin:$PATH >> .bashrc
RUN echo export PATH=${CONDA_HOME}/bin:$PATH >> .profile
RUN ${CONDA} init --all
RUN echo conda activate supernnova >> .bashrc

##################################################################################
# zsh supposedly doesn't work with gpu images so the following is disabled for now
##################################################################################
#
# ####################################
# # Set up locale to avoid zsh errors
# ####################################
# RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
#     && locale-gen --purge --lang en_US \
#     && locale-gen
#
# ###################
# # Set up Oh My Zsh
# ###################
# COPY env/zshrc ${HOME}/.zshrc
# RUN git clone https://github.com/robbyrussell/oh-my-zsh.git ${HOME}/.oh-my-zsh &&\
#     sed -i 's/❯/Docker❯/g' ${HOME}/.oh-my-zsh/themes/refined.zsh-theme
# ENV ZSH=${HOME}/.oh-my-zsh \
#     PATH_TO_COPY=${PATH}

################################################
# Copy and set persmissions of the entry script
################################################
USER root
COPY env/entry_script.sh /
RUN chmod a+rx /entry_script.sh
RUN echo su -m $USERNAME >> /entry_script.sh

###################################
# Set-up the user's starting state
###################################
ENTRYPOINT ["/bin/bash", "/entry_script.sh"]
