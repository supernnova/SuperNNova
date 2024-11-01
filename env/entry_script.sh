#!/bin/bash
set -e

# Activate the Conda environment
# /u/home/miniconda3/bin/conda init --all
# . /u/home/.bashrc
# /u/home/miniconda3/bin/conda activate supernnova

_term() { 
  echo "Caught SIGTERM signal!" 
  kill -TERM "$child" 2>/dev/null
}

trap _term SIGTERM

# Docker build will add lines after this point.  See Dockerfile.
