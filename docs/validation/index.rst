
Validation walkthrough
=========================

Activate the environment
-------------------------------

**Either use docker**

.. code::

    cd env && python launch_docker.py (--use_cuda optional)

**Or activate your conda environment**

.. code::

    source activate <conda_env_name>



Validation
-------------------------------

Assuming a database has been created and models have been trained, a model can be validated as follows:


.. code::

    python run.py --validate_rnn --dump_dir /path/to/dump_dir
    python run.py --validate_rnn --dump_dir /path/to/dump_dir

In that case, the model corresponding to the command line arguments will be loaded and validated. Output will be written in ``dump_dir/models/yourmodelname/``.

Alternatively, one or more model files can be specified

.. code::

    python run.py --validate_rnn --dump_dir /path/to/dump_dir --model_files /path/to/model/file(s)
    python run.py --validate_rnn --dump_dir /path/to/dump_dir --model_files /path/to/model/file(s)

In that case, validation will be carried out for each of the models specified by the model files. This will use the database in ``dump_dir/processed`` directory. 


This will:

- Make predictions on a test set (saved to a file with the ``PRED_`` prefix)
- Compute metrics on the test (saved to a file with the ``METRICS_`` prefix)
- All results are dumped in the same folder as the folder where the trained model was dumped


To make predictions on an independent database than the one used to train a given model

.. code::

    python run.py --dump_dir  /path/to/dump_dir --validate_rnn --model_files path/to/modelfile/modelfile.pt

In this case it will run the model provided in ``model_files`` with the normalization of the model on the database available in ``dump_dir/processed``. Predictions will be saved in ``dump_dir/models/modelname/``.

Predictions format
~~~~~~~~~~~~~~~~~~~~~
For a binary classification task, predictions files contain the following columns:

.. code::

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


RNN speed
-------------------------------

Run RNN classification speed benchmark as follows

.. code::

    python run.py --data --dump_dir /path/to/dump_dir  # create database
    python run.py --speed --dump_dir /path/to/dump_dir

This will create ``tests/dump/stats/rnn_speed.csv`` showing the classification throughput of RNN models.


Calibration
-------------------------------

Assuming a database has been created and models have been trained, evaluate classifier calibration as follows:

.. code::

    python run.py --calibration --dump_dir /path/to/dump_dir --metric_files /path/to/metric_file

This will output a figure in ``path/to/dump_dir/figures`` showing how well a given model is calibrated.
A metric file looks like this: ``METRICS_{model_name}.pickle``. For instance: ``METRICS_DES_vanilla_S_0_CLF_2_R_None_saltfit_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean.pickle``
Multiple metric files can be specified, the results will be charted on the same graph.


Science plots
-------------------------------

Assuming a database has been created and models have been trained, how some graphs of scientific interest:

.. code::

    python run.py --science_plots --dump_dir /path/to/dump_dir --prediction_files /path/to/prediction_file

This will output figures in ``path/to/dump_dir/figures`` showing various plots of interest: Hubble residuals, purity vs redshift etc.
A prediction file looks like this: ``PRED_{model_name}.pickle``. For instance: ``PRED_DES_vanilla_S_0_CLF_2_R_None_saltfit_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean.pickle``


Performance metrics
-------------------------------

Assuming a database has been created and models have been trained, compute performance metrics

.. code::

    python run.py --performance --dump_dir /path/to/dump_dir

This will output a csv file in ``path/to/dump_dir/stats``, which aggregates various performance metrics for each model that has been trained and for which a ``METRICS`` file has been created.
