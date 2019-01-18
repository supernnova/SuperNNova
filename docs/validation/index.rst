
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
~~~~~~~~~~~~~~~

Assuming a database has been created and models have been trained, a model can be validated as follows:


.. code::

    python run.py --validate_rnn --dump_dir /path/to/dump_dir
    python run.py --validate_rnn --dump_dir /path/to/dump_dir

In that case, the model corresponding to the command line arguments will be loaded and validated.

Alternatively, one or more model files can be specified

.. code::

    python run.py --validate_rnn --dump_dir /path/to/dump_dir --model_files /path/to/model/file(s)
    python run.py --validate_rnn --dump_dir /path/to/dump_dir --model_files /path/to/model/file(s)

In that case, validation will be carried out for each of the models specified by the model files.


This will:

- Make predictions on a test set (saved to a file with the ``PRED_`` prefix)
- Compute metrics on the test (saved to a file with the ``METRICS_`` prefix)
- All results are dumped in the same folder as the folder where the trained model was dumped


RNN speed
~~~~~~~~~~

Run RNN classification speed benchmark as follows

.. code::

    python run.py --data --dump_dir /path/to/dump_dir  # create database
    python run.py --speed --dump_dir /path/to/dump_dir

This will create ``tests/dump/stats/rnn_speed.csv`` showing the classification throughput of RNN models.


Calibration
~~~~~~~~~~~~~~

Assuming a database has been created and models have been trained, evaluate classifier calibration as follows:

.. code::

    python run.py --calibration --dump_dir /path/to/dump_dir --metric_files /path/to/metric_file

This will output a figure in ``path/to/dump_dir/figures`` showing how well a given model is calibrated.
A metric file looks like this: ``METRICS_{model_name}.pickle``. For instance: ``METRICS_DES_vanilla_S_0_CLF_2_R_None_saltfit_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean.pickle``
Multiple metric files can be specified, the results will be charted on the same graph.


Science plots
~~~~~~~~~~~~~~

Assuming a database has been created and models have been trained, how some graphs of scientific interest:

.. code::

    python run.py --science_plots --dump_dir /path/to/dump_dir --prediction_files /path/to/prediction_file

This will output figures in ``path/to/dump_dir/figures`` showing various plots of interest: Hubble residuals, purity vs redshift etc.
A prediction file looks like this: ``PRED_{model_name}.pickle``. For instance: ``PRED_DES_vanilla_S_0_CLF_2_R_None_saltfit_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean.pickle``


Performance metrics
~~~~~~~~~~~~~~~~~~~~~

Assuming a database has been created and models have been trained, compute performance metrics

.. code::

    python run.py --performance --dump_dir /path/to/dump_dir

This will output a csv file in ``path/to/dump_dir/stats``, which aggregates various performance metrics for each model that has been trained and for which a ``METRICS`` file has been created.
