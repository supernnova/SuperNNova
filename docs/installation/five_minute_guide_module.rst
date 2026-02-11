
.. _Start_module:

Quickstart guide (pip)
========================

Welcome to SuperNNova! This is a quick start guide so you can start testing our framework. This guide assumes you have installed it with pip, if you want to use the GitHub cloning please refer to :ref:`Start`.

Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pip install
-----------------------------

.. code::

	pip install supernnova

Please beware that SuperNNova only runs properly in Unix systems. 

Setup your environment. 3 options
-----------------------------------

	a) Create a conda virtual env :ref:`CondaConfigurations` (preferred).
	b) Create a docker image: :ref:`DockerConfigurations` .
	c) Install packages manually. Inspect ``env/conda_env.yml`` (or ``env/conda_gpu_env.yml`` when using cuda) and ``pyproject.toml`` for the list of packages we use.

Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For quick tests, a database that contains a limited number of light-curves is provided. It is located in ``tests/raw``. For more information on the available data, check :ref:`DataStructure`. An example of running as module can be found in ``sandbox/example_run_moduler_snn.py``.

Build the database
-----------------------

In the parent folder, you can launch python or ipython with the following:

.. code-block:: python

	import supernnova.conf as conf
	from supernnova.data import make_dataset

	# get config args
	command_arg = "make_data"
	args = conf.get_args(command_arg)

	# create database
	args.dump_dir = "tests/dump"		# conf: where the dataset will be saved
	args.raw_dir = "tests/raw"		# conf: where raw photometry files are saved 
	args.fits_dir = "tests/fits"		# conf: where salt2fits are saved 
	settings = conf.get_settings(command_arg, args)	# conf: set settings
	make_dataset.make_dataset(settings)	# make dataset


Train an RNN
---------------------------------------

.. code-block:: python

	import supernnova.conf as conf
	from supernnova.training import train_rnn

	# get config args
	command_arg = "train_rnn"
	args = conf.get_args(command_arg)

	# train rnn
	args.dump_dir = "tests/dump"		# conf: where the dataset is saved
	args.nb_epoch = 2			# conf: training epochs
	settings = conf.get_settings(command_arg, args)	# conf: set settings
	train_rnn.train(settings)		# train rnn

Validate an RNN
---------------------------------------

.. code-block:: python

	import supernnova.conf as conf
	from supernnova.validation import validate_rnn

	# get config args
	command_arg = "validate_rnn"
	args = conf.get_args(command_arg)

	# validate rnn
	args.dump_dir = "tests/dump"			# conf: where the dataset is saved
	settings = conf.get_settings(command_arg, args)		# conf: set settings
	validate_rnn.get_predictions(settings)		# classify test set

