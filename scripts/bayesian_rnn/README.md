# Language Modelling with Bayesian Recurrent Networks

## Overview

This is a reproduction of [Bayesian Recurrent Netwroks](https://arxiv.org/pdf/1704.02798.pdf) from Fortunato et al.

Specifically, we reproduce the results from Table 1, row `Bayesian RNN (BRNN)`.

## Usage

From this project's root, call

    PYTHONPATH=$PWD:$PYTHONPATH python scripts/mc_dropout/lm_rnn_gal.py  # on CPU, very slow
    PYTHONPATH=$PWD:$PYTHONPATH python scripts/mc_dropout/lm_rnn_gal.py --cuda  # if you have a GPU


## Results

- Obtained on GTX 1080 Ti, pytorch 1.3.1
- Training took 89 minutes
- Test set perplexity 76.82

