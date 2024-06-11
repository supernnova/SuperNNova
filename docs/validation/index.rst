
Validation walkthrough
=========================

Activate the environment
-------------------------------

**Either use docker**

.. code-block:: bash

    cd env && python launch_docker.py (--use_cuda optional)

**Or activate your conda environment**

.. code-block:: bash

    source activate <conda_env_name>



Validation an RNN model
-------------------------------
To validate an RNN model, you can use ``snn validate_rnn`` with valid options:

.. code-block:: bash

    snn validate_rnn [option]

A list of valid options can be shown by using the ``--help`` flag:

.. code-block:: bash

    snn validate_rnn --help

.. code-block:: none

    usage: snn validate_rnn [options]

    optional arguments:
    --calibration                  Plot calibration of trained classifiers
    --config_file                  YML config file
    --dump_dir                     Default path where data and models are dumped
    --help                         Show custom help message
    --model_files                  Path to model files
    ... ...


Assuming a database has been created (see :ref:`DataStructure`) and models have been trained (see :ref:`TrainRnn`), a model can be validated as follows:

.. code-block:: bash

    snn validate_rnn --dump_dir /path/to/dump_dir


In that case, the model corresponding to the command line arguments will be loaded and validated. Output will be written in ``dump_dir/models/yourmodelname/``.

Alternatively, one or more model files can be specified

.. code-block:: bash

    snn validate_rnn --dump_dir /path/to/dump_dir --model_files /path/to/model/file(s)
    
In that case, validation will be carried out for each of the models specified by the model files. This will use the database in ``dump_dir/processed`` directory. 


This will:

- Make predictions on a test set (saved to a file with the ``PRED_`` prefix)
- Compute metrics on the test (saved to a file with the ``METRICS_`` prefix)
- All results are dumped in the same folder as the folder where the trained model was dumped


To make predictions on an independent database than the one used to train a given model

.. code-block:: bash

    snn validate_rnn --dump_dir /path/to/dump_dir --model_files path/to/modelfile/modelfile.pt

In this case it will run the model provided in ``model_files`` with the features and normalization of the model on the database available in ``dump_dir/processed``. Predictions will be saved in ``dump_dir/models/modelname/``. If uncertain about the model features, take a look at the ``cli_args.json`` in the model directory.

Predictions format
~~~~~~~~~~~~~~~~~~~~~
For a binary classification task, predictions files contain the following columns:

.. code-block:: none

    all_class0            float32  - probability of classifying complete light-curves as --sntype [0] (usually Ia)
    all_class1            float32  - probability of classifying complete light-curves as --sntype [1:] (usually nonIas)
    PEAKMJD-2_class0      float32  - probability of classifying light-curves up to 2 days before maximum as --sntype [0] (usually Ia)
    PEAKMJD-2_class1      float32  - probability of classifying light-curves up to 2 days before maximum as  --sntype [1:] (usually nonIas)
    PEAKMJD-1_class0      float32  - up to one day before maximum light
    PEAKMJD-1_class1      float32
    PEAKMJD_class0        float32  - up to maximum light lightcurves
    PEAKMJD_class1        float32
    PEAKMJD+1_class0      float32  - one day post maximum lightcurves
    PEAKMJD+1_class1      float32
    PEAKMJD+2_class0      float32  - two days post maximum lightcurves
    PEAKMJD+2_class1      float32
    all_random_class0     float32  - Out-of-distribution: probability of classifying randomly generated complete lightcurves as --sntype [0]
    all_random_class1     float32
    all_reverse_class0    float32  - Out-of-distribution: probability of classifying time reversed complete lightcurves as --sntype [0]
    all_reverse_class1    float32
    all_shuffle_class0    float32  - Out-of-distribution: probability of classifying shuffled complete lightcurves (permutations of time-series) as --sntype [0]
    all_shuffle_class1    float32
    all_sin_class0        float32  - Out-of-distribution: probability of classifying sinusoidal complete lightcurves (permutations of time-series) as --sntype [0]
    all_sin_class1        float32
    target                  int64  - Type of the supernova, simulated class.
    SNID                    int64  - ID number of the light-curve

these columns rely on maximum light information and target (original type) from simulations. Out-of-distribution classifications are done on the fly. Bayesian Networks (variational and Bayes by Backprop) have an entry for each probability distribution sampling, to get the mean and std of the classification read the ``_aggregated.pickle`` file.

You can also use a YAML file to specify option arguments. Please see :ref:`UseYaml` for more information.

RNN speed
-------------------------------

Run RNN classification speed benchmark as follows

.. code-block:: bash

    snn make_data --dump_dir /path/to/dump_dir --raw_dir tests/raw # create database
    snn validate_rnn --speed --dump_dir /path/to/dump_dir

This will create ``tests/dump/stats/rnn_speed.csv`` showing the classification throughput of RNN models.


.. _ValidateCalibration:

Calibration
-------------------------------

Assuming a database has been created and models have been trained, evaluate classifier calibration as follows:

.. code-block:: bash

    snn validate_rnn --calibration --dump_dir /path/to/dump_dir 

This will output a figure in ``path/to/dump_dir/figures`` showing how well a given model is calibrated.
