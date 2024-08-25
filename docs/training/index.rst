.. _TrainRnn:

Training walkthrough
=========================

Activate the environment
-------------------------------

**Either use docker**

.. code-block:: bash

    cd env && python launch_docker.py (--use_cuda optional)

**Or activate your conda environment**

.. code-block:: bash

    source activate <conda_env_name>


Training an RNN model
-------------------------------
To train an RNN model, you can use ``snn train_rnn`` with valid options:

.. code-block:: bash

    snn train_rnn [option]

A list of valid options can be shown by using the ``--help`` flag:

.. code-block:: bash

    snn train_rnn --help

.. code-block:: none

    usage: snn train_rnn [options]

    optional arguments:
    --additional_train_var         Additional training variables
    --batch_size                   Batch size
    --bidirectional                Use bidirectional models
    --calibration                  Plot calibration of trained classifiers
    --config_file                  YML config file
    --cyclic                       Use cyclic learning rate
    ... ...


Assuming a database has been created (see :ref:`DataStructure`), you can train an RNN model as follows:

.. code-block:: bash

    snn train_rnn --dump_dir /path/to/your/dump/dir # train and validate


This will:

- Train an RNN classifier
- All outputs are dumped to ``/path/to/your/dump/dir/models/vanilla_S_0_CLF_2_R_None_saltfit_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean``
- Save the trained classifier: ``vanilla_S_0_CLF_2_R_None_saltfit_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean.pt``
- Make predictions on a test set: ``PRED_DES_vanilla_S_0_CLF_2_R_None_saltfit_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean.pt.pickle``
- Compute metrics on the test: ``METRICS_DES_vanilla_S_0_CLF_2_R_None_saltfit_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean.pt.pickle``
- Save loss curves: ``train_and_val_loss_vanilla_S_0_CLF_2_R_None_saltfit_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean.png``
- Save training statistics: ``training_log.json``

You can also use a YAML file to specify option arguments. Please see :ref:`UseYaml` for more information.

Training an RNN model with different normalizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The data for training and validation can be normalized for better performance. Currrently the options for ``--norm`` are ``none, global, perfilter, cosmo, cosmo_quantile``. The default normalization is ``global``.

For ``global, perfilter`` normalizations, features (f) are first log transformed and then scaled. The log transform (fl) uses the minimum value of the feature min(f) and a constant (epsilon) to center the distribution in zero as follows: fl = log (−min( f ) + f + epsilon). Using the mean and standard deviation of the log transform (mu,sigma(fl)), standard scaling is applied: fˆ = ( fl − mu( fl))/sigma( fl). In the “global” scheme, the minimum, mean and standard deviation are computed over all fluxes (resp. all errors). In the “per-filter” scheme, they are computed for each filter.

When using ``--redshift`` for classification, we suggest to use either ``cosmo,cosmo_quantile`` norms. These normalizations blur the distance information that SNe Ia provide with apparent flux which together with redshift information may bias the classification for cosmology. For this, light-curves are normalized to a flux ~1 using either the maximum flux at any filter (``cosmo``) or the 99 quantile of the flux distribution (``cosmo_quantile``). The latter is mroe robust against outliers.

Training an RNN model with Stochastic Weight Averaging Gaussian (SWAG) enabled
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Stochastic Weight Averaging Gaussian (SWAG) is a widely used technique for approximate Bayesian inference in deep learning. It was introduced in the paper `A Simple Baseline for Bayesian Uncertainty in Deep Learning <https://arxiv.org/abs/1902.02476>`_ by Wesley Maddox, Timur Garipov, Pavel Izmailov, Dmitry Vetrov, and Andrew Gordon Wilson.

This technique is implementd in **supernnova** (version xxx). You can enable SWAG during training with the ``--swag`` flag:

.. code-block:: bash

    snn train_rnn --dump_dir /path/to/your/dump/dir --swag

This will generate all the files for standard RNN model training described above, with the following addition items:

- SWAG model: ``vanilla_*_swag.pt``

- SWA prediction: ``PRED_vanilla_*_swa.pickle``

- SWAG prediction: ``PRED_vanilla_*_scale_0.5_cov_swag.pickle``

- SWAG prediction aggregated: ``PRED_vanilla_*_scale_0.5_cov_swag_aggregated.pickle``

SWAG Configuration Options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Starting Epoch: SWAG typically begins towards the end of training, with a default start at Epoch ``83``. You can adjust this with the ``--swag_start_epoch`` flag.

- Number of Samples: The number of samples to draw during validation is controlled by the ``--swag_samples`` flag. The default is ``30``.

- Scaling Parameter: The scaling parameter for the covariance is set using the ``--swag_scale`` flag, with a default value of ``0.5``, as recommended in the original paper. Setting the scale to ``0`` disables covariance calculation, effectively reducing SWAG to standard Stochastic Weight Averaging (SWA).

- Covariance Calculation: If you wish to disable the calculation of low-rank covariance, use the ``--swag_no_cov`` flag.
