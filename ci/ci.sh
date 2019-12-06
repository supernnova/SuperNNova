#!/bin/sh

set -eux

DIR=$(cd "$(dirname "$0")"; pwd -P)

DUMP_DIR="/tmp/snndump"
mkdir -p "$DUMP_DIR"

docker run -it --rm --user $(id -u) -v "$DIR/..":/home/SuperNNova \
    -v "$DUMP_DIR":/home/snndump -- rnn-cpu:latest -lc "SuperNNova/ci/train.sh"
