
![Logo](docs/SuperNNova.png)

Read the documentation at [https://supernnova.readthedocs.io](https://supernnova.readthedocs.io/en/latest/)

## Repository overview

    ├── supernnova              --> main module
        ├──data                 --> scripts to create the processed database
        ├──visualization        --> data plotting scripts
        ├──training             --> training scripts
        ├──validation           --> validation scripts
        ├──utils                --> utilities used throughout the module
    ├── tests                   --> unit tests to check data processing
    ├── sandbox                 --> WIP scripts


## Minimum instructions for end to end data creation, training and plotting on a toy dataset

    python run.py --data  --dump_dir tests/dump
    python run.py --train_rnn --dump_dir tests/dump
    python run.py --train_rnn --model variational --dump_dir tests/dump
    python run.py --train_rnn --model bayesian --dump_dir tests/dump
    python run.py --train_rf --dump_dir tests/dump

## Reproduce (soon to be published) results

    python run_paper.py

## Pipeline description

- Parse raw data in FITS format
- Create processed database in HDF5 format
- Train Recurrent Neural Networks (RNN) or Random Forests (RF) to classify photometric lightcurves
- Validate on test set


## Running tests with py.test

    PYTHONPATH=$PWD:$PYTHONPATH pytest -W ignore --cov supernnova tests


## Build docker images

    cd docker
    make cpu TAG=cpu  # cpu image
    make gpu TAG=gpu  # gpu image

## Build docs

    cd docs && make clean && make html && cd ..
    firefox docs/_build/html/index.html