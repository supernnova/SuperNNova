#!/bin/sh

set -eux

export PATH="/home/miniconda3/bin:$PATH"

DIR=$(cd "$(dirname "$0")"; pwd -P)

cd $DIR/..

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
