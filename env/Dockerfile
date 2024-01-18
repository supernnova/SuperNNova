#####################################
# Default build type is defined here
#####################################
ARG TARGET="cpu"

##############################
# Base image for 'cpu' builds
##############################
FROM ubuntu:22.04 as cpu
RUN apt-get update -y && \
    apt-get upgrade -y && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends\
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
       wget \
       unzip \
       locales \
       python3.11 python3-distutils python3-pip python3-apt && \
    rm -rf /var/lib/apt/lists/*

################################################
# Base image for 'gpu9' builds, using CUDA v9.0
################################################
FROM nvcr.io/nvidia/cuda:9.0-devel-ubuntu16.04 as gpu9
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install ca-certificates -y
RUN apt-get update -y && \
    apt-get upgrade -y && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends\
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
       wget \
       unzip \
       locales && \
    rm -rf /var/lib/apt/lists/*

#####################################################
# Build and install Python
#----------------------------------------------------
# Needs to be done this way beacause modern versions
# of Python do not have apt-get registry entries for
# some of the old Ubuntu versions we wish to support
#####################################################
ENV PYTHON_VERSION=3.11.7
RUN cd /opt && \
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar xzvf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make && \
    make install

###########################################################
# Base image for 'gpu' builds, using a recent CUDA version
#----------------------------------------------------------
# Note that Nvidia no longer supports a 'latest' tag,
# so this needs to be maintained
###########################################################
FROM nvcr.io/nvidia/cuda:12.3.1-devel-ubuntu22.04 as gpu
RUN apt-get update -y && \
    apt-get upgrade -y && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends\
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
       wget \
       unzip \
       locales \
       python3.11 python3-distutils python3-pip python3-apt && \
    rm -rf /var/lib/apt/lists/*

#####################################################
# Everything that follows is the same for all builds
#####################################################
FROM ${TARGET}

WORKDIR /u/home
ENV HOME=/u/home

###########
# Anaconda
###########
RUN curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /home/miniconda3 && \
    rm ~/miniconda.sh && \
    /home/miniconda3/bin/conda config --add channels conda-forge && \
    /home/miniconda3/bin/conda clean -ya 

#################
# Install Poetry
#################
RUN pip install poetry

#####################################################
# Copy and install application & Python dependencies
#####################################################
COPY python .
COPY pyproject.toml .
COPY poetry.lock .
RUN poetry install --all-extras

####################################
# Set up locale to avoid zsh errors
####################################
RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
    && locale-gen --purge --lang en_US \
    && locale-gen

#########################################################
# Set up  environment variables needed for libs installs
#########################################################
ENV PATH=$PATH:/u/home/.local/bin
ENV TMP_PATH=/u/home/dump \
    LD_LIBRARY_PATH="/usr/local/lib:/home/miniconda3/lib:$LD_LIBRARY_PATH" \
    PATH=/home/miniconda3/bin:$PATH \
    LANG=en_US.utf8 \
    USER=researcher \
    USER_ID=2018 \
    USER_GID=2018

##################################################################
# Add a user which will be mapped to real user running the docker
##################################################################
RUN mkdir -p /u/home && \
    groupadd --gid "${USER_GID}" "$USER" && \
    useradd \
    --uid ${USER_ID} \
    --gid ${USER_GID} \
    --home-dir /u/home \
    --shell /usr/bin/zsh \
    ${USER}

###################
# Set up Oh My Zsh
###################
COPY env/zshrc /u/home/.zshrc
RUN git clone https://github.com/robbyrussell/oh-my-zsh.git /u/home/.oh-my-zsh &&\
    sed -i 's/❯/Docker❯/g' /u/home/.oh-my-zsh/themes/refined.zsh-theme
ENV ZSH=/u/home/.oh-my-zsh \
    PATH_TO_COPY=${PATH}

COPY env/entry_script.sh /
RUN chmod u+x /entry_script.sh

ENTRYPOINT ["/bin/bash", "/entry_script.sh"]
