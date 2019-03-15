
.. _Start_module:

Quickstart guide (pip)
========================

Welcome to SuperNNova! This is a quick start guide so you can start testing our framework. THis guide assumes you have installed it with pip, if you want to use the GitHub cloning please refer to :ref:`Start`.

Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pip install
-----------------------------

.. code::

	pip install supernnova

Setup your environment. 3 options
-----------------------------------

	a) Create a conda virtual env :ref:`CondaConfigurations` (preferred).
	b) Create a docker image: :ref:`DockerConfigurations` .
	c) Install packages manually. Inspect ``conda_env.txt`` for the list of packages we use.

Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For quick tests, a database that contains a limited number of light-curves is provided. It is located in ``tests/raw``. For more information on the available data, check :ref:`DataStructure`. An example of running as module can be found in ``sandbox/example_run_moduler_snn.py``.

Build the database
-----------------------

In the parent folder, where ``run.py`` is located you can launch python or ipython with the following:

.. code::

	import supernnova.conf as conf
	from supernnova.data import make_dataset

	# get config args
	args =  conf.get_args()

	# create database
	args.data = True			# conf: making new dataset
	args.dump_dir = "tests/dump"		# conf: where the dataset will be saved
	args.raw_dir = "tests/raw"		# conf: where raw photometry files are saved 
	args.fits_dir = "tests/fits"		# conf: where salt2fits are saved 
	settings = conf.get_settings(args)	# conf: set settings
	make_dataset.make_dataset(settings)	# make dataset


Train an RNN
---------------------------------------

.. code::

	import supernnova.conf as conf
	from supernnova.training import train_rnn

	# get config args
	args =  conf.get_args()

	args.train_rnn = True			# conf: train rnn
	args.dump_dir = "tests/dump"		# conf: where the dataset is saved
	args.nb_epoch = 2			# conf: training epochs
	settings = conf.get_settings(args)	# conf: set settings
	train_rnn.train(settings)		# train rnn

Validate an RNN
---------------------------------------

.. code::

	import supernnova.conf as conf
	from supernnova.validation import validate_rnn

	# get config args
	args =  conf.get_args()

	args.validate_rnn = False			# conf: validate rnn
	args.dump_dir = "tests/dump"			# conf: where the dataset is saved
	settings = conf.get_settings(args)		# conf: set settings
	validate_rnn.get_predictions(settings)		# classify test set

