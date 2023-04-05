#!/bin/bash

# local testing
source activate snn

echo 'Simple data'
python run.py --data --raw_dir tests/raw --dump_dir tests/dump
python run.py --data --raw_dir tests/raw_csv --dump_dir tests/dump_csv
echo 'ok'

echo 'Simple model'
python run.py --dump_dir tests/dump --train_rnn --nb_epoch 1
python run.py --dump_dir tests/dump_csv --train_rnn --nb_epoch 1
echo 'ok'

echo 'Photo window'
python run.py --data --raw_dir tests/raw --dump_dir tests/dump_window --photo_window_files HEAD --photo_window_var PEAKMJD
python run.py --data --raw_dir tests/raw_csv --dump_dir tests/dump_window_csv --photo_window_files HEAD --photo_window_var PEAKMJD
echo 'ok'
