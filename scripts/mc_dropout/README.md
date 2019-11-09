# Language Modelling with Variational Monte Carlo dropout

## Overview

This is a reproduction of [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](https://arxiv.org/abs/1512.05287) from Yarin Gal and Zoubin Ghahramani.

Specifically, we reproduce the results from Table 1, row `Variational (untied weights)`.

## Usage

From this project's root, call

    PYTHONPATH=$PWD:$PYTHONPATH python scripts/mc_dropout/lm_rnn_gal.py  # on CPU, very slow
    PYTHONPATH=$PWD:$PYTHONPATH python scripts/mc_dropout/lm_rnn_gal.py --cuda  # if you have a GPU


## Results

- Obtained on GTX 1080 Ti, pytorch 1.3.1
- Training took 17 minutes
- Test set perplexity 78.26

