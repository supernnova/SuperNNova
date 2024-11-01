
.. _Start:

Quickstart guide (GitHub)
============================

Welcome to SuperNNova!

This is a quick start guide so you can start testing our framework. 
If you want to install SuperNNova as a module, please take a look at :ref:`Start_module`.


Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone the GitHub repository
-----------------------------

.. code::

	git clone https://github.com/supernnova/supernnova.git

Setup your environment. 3 options
-----------------------------------

Please beware that SuperNNova only runs properly in Unix systems (Linux, MacOS). 
	a) Create a docker image: :ref:`DockerConfigurations` .
	b) Create a conda virtual env :ref:`CondaConfigurations` .
	c) Install packages manually. Inspect ``env/conda_env.yml`` (or ``env/conda_gpu_env.yml`` when using cuda) and ``pyproject.toml`` for the list of packages we use.

Verify installation 
-----------------------------------
This package provides its own bash command ``snn``. Once the installation is completed successfully, you should be able to run the following line in the terminal:

.. code-block:: bash

    snn --help

.. code-block:: none

    Usage: snn <command> <options> <arguments>

    Available commands:

        make_data        create dataset for ML training
        train_rnn        train RNN model
        validate_rnn     validate RNN model
        show             vitualize different types of plot
        performance      get method performance and paper plots

    Type snn <command> --help for usage help on a specific command.
    For example, snn make_data --help will list all data creation options.    

where ``make_data``, ``train_rnn``, ``validate_rnn``, ``show`` and ``performance`` are the sub-commands. 

Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For quick tests, a database that contains a limited number of light-curves is provided. It is located in ``tests/raw``. For more information on the available data, check :ref:`DataStructure`.

Using command line 
-----------------------
Build the database

.. code-block:: bash

    snn make_data --dump_dir tests/dump --raw_dir tests/raw

an additional argument ``--fits_dir tests/fits`` can provide a SALT2 fits file for random forest training and interpretation.


Train an RNN

.. code-block:: bash

    snn train_rnn --dump_dir tests/dump

With this command you are training and validating our Baseline RNN with the test database and generating test lightcurves as well. The trained model will be saved in a newly created model folder inside ``tests/dump/models``.

The model folder has been named as follows: ``vanilla_S_0_CLF_2_R_None_photometry_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean`` (See below for the naming conventions). This folder's contents are:

- **saved model** (``*.pt``): PyTorch RNN model.

- **statistics** (``METRICS*.pickle``): pickled Pandas DataFrame with accuracy and other performance statistics for this model.

- **predictions** (``PRED*.pickle``): pickled Pandas DataFrame with the predictions of our model on the test set.

- **figures** (``train_and_val_*.png``): figures showing the evolution of the chosen metric at each training step.

Remember that our data is split in training, validation and test sets.

The test light-curves and their predictions can be inspected in ``tests/dump/lightcurves``

**You have trained, validated and tested your model.**

.. _UseYaml:

Using Yaml
-----------------------
You can also save arguments of options in an YAML file, and load it:

.. code-block:: bash

    snn <command> --config_file <yaml file>

Example YAML files can be found in the folder ``configs_yml``, where ``classify.yml`` is an example of classification using existing model.

**Notice**: you can include options for different sub-commands in the same YAML file. 

Build the database

.. code-block:: bash

    snn make_data --config_file configs_yml/default.yml

Train an RNN

.. code-block:: bash

    snn train_rnn --config_file configs_yml/default.yml 


You can also update option specified in the YAML file by using command-line option:

.. code-block:: bash

    snn make_data --config_file configs_yml/simple.yml --dump_dir tests/dump2
    # or
    snn make_data --dump_dir tests/dump2 --config_file configs_yml/simple.yml

The data will be dumpped to ``tests/dump2`` instead of ``tests/dump`` specified in ``config_yml/simple.yml``.

**Notice**: adding command-line options will update the arguments at runtime, not change the YAML file itself. 






Reproduce SuperNNova paper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To reproduce the results of the paper please use the branch ``paper`` and run:

.. code::

    cd SuperNNova && python run_paper.py --debug --dump_dir tests/dump

``--debug``  will train simplified models with a reduced number of epochs. Remove this flag for full reproducibility.
With the ``--debug`` flag on, this should take between 15 and 30 minutes on the CPU.


Naming conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **vanilla/variational/bayesian**: The type of RNN to be trained. ``variational`` and ``bayesian`` are bayesian recurrent networks

- **S_0**: seed used for training. Default is 0.

- **CLF_2**: number of targets to be used in classification. This case uses two classes: type Ia supernovae vs. all others.

- **R_None**: host-galaxy redshift provided. Options: ``zpho`` (photometric) or ``zspe`` (spectroscopic)

- **photometry**: data used. In our database we split light-curves that have a succesful SALT2 fit (``saltfit``) and the complete dataset (``photometry``).

- **DF_1.0**: data fraction used in training. With large datasets it is usefult to test training with a fraction of the available training set. In this case we use the whole dataset (``1.0``).

- **N_global**: normalization used. Default: ``global``.

- **lstm**: type of layer used. Default ``lstm``.

- **32x2**: hidden layer dimension x number the layers.

- **0.05**: dropout value.

- **128**: batch size.

- **True**: if this model is bidirectional.

- **mean**: output option. ``mean`` is mean pooling.

The naming convention is defined in ``SuperNNova/conf.py``.
