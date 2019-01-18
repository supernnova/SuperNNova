
.. _Start:

Quickstart start guide
========================

Welcome to SuperNNova!

This is a quick start guide so you can start testing our framework.

Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone the GitHub repository
-----------------------------

.. code::

	git clone https://gitlab.com/tdeboissiere/supernnova.git

Setup your environment. 3 options
-----------------------------------


	a) Create a docker image: :ref:`DockerConfigurations` .
	b) Create a conda virtual env :ref:`CondaConfigurations` .
	c) Install packages manually. Inspect ``conda_env.txt`` for the list of packages we use.

Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For quick tests, a database that contains a limited number of light-curves is provided. It is located in ``tests/dump/raw``. For more information on the available data, check :ref:`DataStructure`.

Build the database
-----------------------

.. code::

    python run.py --data --dump_dir tests/dump

Train an RNN
---------------------------------------


.. code::

    $ python run.py --train_rnn --dump_dir tests/dump

With this command you are training and validating our Baseline RNN with the test database. The trained model will be saved in a newly created model folder inside ``tests/dump/models``.

The model folder has been named as follows: ``vanilla_S_0_CLF_2_R_None_saltfit_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean`` (See below for the naming conventions). This folder's contents are:

- **saved model** (``*.pt``): PyTorch RNN model.

- **statistics** (``METRICS*.pickle``): pickled Pandas DataFrame with accuracy and other performance statistics for this model.

- **predictions** (``PRED*.pickle``): pickled Pandas DataFrame with the predicitons of our model on the test set.

- **figures** (``train_and_val_*.png``): figures showing the evolution of the chosen metric at each training step.

Remember that our data is split in training, validation and test sets.

**You have trained, validated and tested your model.** You can now inspect the test light-curves and their predictions in ``tests/dump/lightcurves``.


Reproduce SuperNNova paper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To reproduce the results of the paper:

.. code::

    $ cd SuperNNova && python run_paper.py --debug --dump_dir tests/dump

``--debug``  will train simplified models with a reduced number of epochs. Remove this flag for full reproducibility.
With the ``--debug`` flag on, this should take between 15 and 30 minutes on the CPU.


Naming conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **vanilla/variational/bayesian**: The type of RNN to be trained. ``variational`` and ``bayesian`` are bayesian recurrent networks

- **S_0**: seed used for training. Default is 0.

- **CLF_2**: number of targets to be used in classification. This case uses two classes: type Ia supernovae vs. all others.

- **R_None**: host-galaxy redshift provided. Options: ``zpho`` (photometric) or ``zspe`` (spectroscopic)

- **saltfit**: data used. In our database we split light-curves that have a succesful SALT2 fit (``saltfit``) and the complete dataset (``photometry``).

- **DF_1.0**: data fraction used in training. With large datasets it is usefult to test training with a fraction of the available training set. In this case we use the whole dataset (``1.0``).

- **N_global**: normalization used. Default: ``global``.

- **lstm**: type of layer used. Default ``lstm``.

- **32x2**: hidden layer dimension x number the layers.

- **0.05**: dropout value.

- **128**: batch size.

- **True**: if this model is bidirectional.

- **mean**: output option. ``mean`` is mean pooling.

The naming convention is defined in ``SuperNNova/conf.py``.