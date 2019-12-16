
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
- dsupport aliases in conf.yml

- input size computation (not hardcoded)  --> will keep hardcoded for now
- data_utils list_training_features to normalize in config --> will keep hardcoded for now


## tests
- runs without salt fit
- can use a photometric time window
- can deal with incomplete type dictionaries
- can deal with two different sntypes for type Ia


## Running tests with py.test <a name="tests"></a>

    PYTHONPATH=$PWD:$PYTHONPATH pytest -W ignore --cov supernnova tests


## Build docs <a name="docs"></a>

    cd docs && make clean && make html && cd ..
    firefox docs/_build/html/index.html


## Run docker

    docker run -it --rm\
    -v </path/to/SuperNNova>:/u/home/SuperNNova \
    -p 8080:8080 \
    -e HOST_USER_ID=$(id -u) \
    -e HOST_USER_GID=$(id -g) rnn-cpu:latest

    docker run -it  --gpus all --rm -p 8080:8080 -v /home/tmain/SuperNNova_refactor:/u/home/SuperNNova -e HOST_USER_ID=$(id -u) -e HOST_USER_GID=$(id -g) rnn-gpu:latest
