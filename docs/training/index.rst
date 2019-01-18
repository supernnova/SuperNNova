
Training walkthrough
=========================



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
