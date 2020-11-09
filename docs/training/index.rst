
Training walkthrough
=========================

Activate the environment
-------------------------------

**Either use docker**

.. code::

    cd env && python launch_docker.py (--use_cuda optional)

**Or activate your conda environment**

.. code::

    source activate <conda_env_name>


Training a randomforest model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code::

    python run.py --data --dump_dir /path/to/your/dump/dir # build the data
    python run.py --train_rf --dump_dir /path/to/your/dump/dir # train and validate

This will:

- Train a randomforest classifier
- All outputs are dumped to ``/path/to/your/dump/dir/models/randomforest_S_0_CLF_2_R_None_saltfit_DF_1.0_N_global``
- Save the trained classifier: ``randomforest_S_0_CLF_2_R_None_saltfit_DF_1.0_N_global.pickle``
- Make predictions on a test set: ``PRED_DES_randomforest_S_0_CLF_2_R_None_saltfit_DF_1.0_N_global.pickle``
- Compute metrics on the test: ``METRICS_DES_randomforest_S_0_CLF_2_R_None_saltfit_DF_1.0_N_global.pickle``


Training an RNN model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code::

    python run.py --data --dump_dir /path/to/your/dump/dir # build the data
    python run.py --train_rnn --dump_dir /path/to/your/dump/dir # train and validate

This will:

- Train an RNN classifier
- All outputs are dumped to ``/path/to/your/dump/dir/models/vanilla_S_0_CLF_2_R_None_saltfit_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean``
- Save the trained classifier: ``vanilla_S_0_CLF_2_R_None_saltfit_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean.pt``
- Make predictions on a test set: ``PRED_DES_vanilla_S_0_CLF_2_R_None_saltfit_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean.pt.pickle``
- Compute metrics on the test: ``METRICS_DES_vanilla_S_0_CLF_2_R_None_saltfit_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean.pt.pickle``
- Save loss curves: ``train_and_val_loss_vanilla_S_0_CLF_2_R_None_saltfit_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean.png``
- Save training statistics: ``training_log.json``

Training an RNN model with different normalizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The data for training and validation can be normalized for better performance. Currrently the options for ``--norm`` are ``none, global, perfilter, cosmo, cosmo_quantile``. The default normalization is ``global``.

For ``global, perfilter`` normalizations, features (f) are first log transformed and then scaled. The log transform (fl) uses the minimum value of the feature min(f) and a constant (epsilon) to center the distribution in zero as follows: fl = log (−min( f ) + f + epsilon). Using the mean and standard deviation of the log transform (mu,sigma(fl)), standard scaling is applied: fˆ = ( fl − mu( fl))/sigma( fl). In the “global” scheme, the minimum, mean and standard deviation are computed over all fluxes (resp. all errors). In the “per-filter” scheme, they are computed for each filter.

When using ``--redshift`` for classification, we suggest to use either ``cosmo,cosmo_quantile`` norms. These normalizations blur the distance information that SNe Ia provide with apparent flux which together with redshift information may bias the classification for cosmology. For this, light-curves are normalized to a flux ~1 using either the maximum flux at any filter (``cosmo``) or the 99 quantile of the flux distribution (``cosmo_quantile``). The latter is mroe robust against outliers.