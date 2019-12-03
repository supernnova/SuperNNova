
![Logo](docs/SuperNNova.png)

### How to
PYTHONPATH=$PWD:$PYTHONPATH python scripts/peak/data.py scripts/peak/conf.yml
PYTHONPATH=$PWD:$PYTHONPATH python scripts/peak/rnn.py scripts/peak/conf.yml

# env
conda update -y conda
# mac os
conda install pytorch torchvision -c pytorch
# linux
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# all
pip install pyyaml
conda config --add channels conda-forge
pip install \
    h5py \
    matplotlib \
    colorama \
    tqdm \
    scipy \
    natsort \
    pandas \
    astropy \
    ipdb \
    scikit-learn \
    pytest \
    unidecode 
pip install sphinx sphinx-autobuild sphinxcontrib-napoleon sphinx_rtd_theme
pip install seaborn pytest-sugar pytest-cov sphinx-argparse tabulate tensorboard

# cleaning up
conda clean -ya

### TO DO

- check MC dropout and Bayes behaviour, in particular, check that sampling multiple predictions for same lightcurve works
- single function to compute loss
- save constants in dump folder?
- input file format conversion in SNN or apart?
- implement different normalizations
- eliminate physics-analysis only features in group_features_list

- input size computation (not hardcoded)  --> will keep hardcoded for now
- data_utils list_training_features to normalize in config --> will keep hardcoded for now

## tests
- runs without salt fit
- can use a photometric time window
- can deal with incomplete type dictionaries
- can deal with two different sntypes for type Ia



### Read the documentation
For the main branch:
[https://supernnova.readthedocs.io](https://supernnova.readthedocs.io/en/latest/)


### Installation
Clone this repository (preferred)
```bash
git clone https://github.com/supernnova/supernnova.git
```
or install pip module (check versioning)
```bash
pip install supernnova
```

### Read the paper preprint

[Paper in ArXiv](https://arxiv.org/abs/1901.06384)
The paper was produced using the branch "paper".


## Table of contents
1. [Repository overview](#overview)
2. [Getting Started](#start)
    1. [With Conda](#conda)
    2. [With Docker](#docker)
3. [Usage](#usage)
3. [Reproduce paper](#paper)
4. [Pipeline Description](#pipeline)
5. [Running tests](#test)
6. [Build the docs](#docs)

## Repository overview <a name="overview"></a>

    ├── supernnova              --> main module
        ├──data                 --> scripts to create the processed database
        ├──visualization        --> data plotting scripts
        ├──training             --> training scripts
        ├──validation           --> validation scripts
        ├──utils                --> utilities used throughout the module
    ├── tests                   --> unit tests to check data processing
    ├── sandbox                 --> WIP scripts

## Getting started <a name="start"></a>

### With Conda <a name="conda"></a>

    cd env

    # Create conda environment
    conda create --name <env> --file <conda_file_of_your_choice>

    # Activate conda environment
    source activate <env>

### With Docker <a name="docker"></a>

    cd env

    # Build docker images
    make cpu  # cpu image
    make gpu  # gpu image (requires NVIDIA Drivers + nvidia-docker)

    # Launch docker container
    docker run -it --rm -v /home/tmain/Lyrebird/SuperNNova:/home/SuperNNova rnn-cpu:latest


For more detailed instructions, check the full [setup instructions](https://supernnova.readthedocs.io/en/latest/installation/python.html)


## Usage <a name="usage"></a>

When cloning this repository:

    # Create data
    python run.py --data  --dump_dir tests/dump --raw_dir tests/raw --fits_dir tests/fits

    # Train a baseline RNN
    python run.py --train_rnn --dump_dir tests/dump

    # Train a variational dropout RNN
    python run.py --train_rnn --model variational --dump_dir tests/dump

    # Train a Bayes By Backprop RNN
    python run.py --train_rnn --model bayesian --dump_dir tests/dump

    # Train a RandomForest
    python run.py --train_rf --dump_dir tests/dump

When using pip, a full example is [https://supernnova.readthedocs.io](https://supernnova.readthedocs.io/en/latest/)

    # Python
    import supernnova.conf as conf
    from supernnova.data import make_dataset

    # get config args
    args =  conf.get_args()

    # create database
    args.data = True            # conf: making new dataset
    args.dump_dir = "tests/dump"        # conf: where the dataset will be saved
    args.raw_dir = "tests/raw"      # conf: where raw photometry files are saved 
    args.fits_dir = "tests/fits"        # conf: where salt2fits are saved 
    settings = conf.get_settings(args)  # conf: set settings
    make_dataset.make_dataset(settings) # make dataset

## Reproduce paper results <a name="paper"></a>
Please change to branch ``paper``:

    python run_paper.py

## General pipeline description <a name="pipeline"></a>

- Parse raw data in FITS format
- Create processed database in HDF5 format
- Train Recurrent Neural Networks (RNN) or Random Forests (RF) to classify photometric lightcurves
- Validate on test set


## Running tests with py.test <a name="tests"></a>

    PYTHONPATH=$PWD:$PYTHONPATH pytest -W ignore --cov supernnova tests


## Build docs <a name="docs"></a>

    cd docs && make clean && make html && cd ..
    firefox docs/_build/html/index.html


## Run docker

    docker run -it --rm\
    -v </path/to/SuperNNova>:/u/home/SuperNNova \
    -e HOST_USER_ID=$(id -u) \
    -e HOST_USER_GID=$(id -g) rnn-cpu:latest

    docker run -it  --gpus all --rm\
    -v </path/to/SuperNNova>:/u/home/SuperNNova \
    -e HOST_USER_ID=$(id -u) \
    -e HOST_USER_GID=$(id -g) rnn-gpu:latest
