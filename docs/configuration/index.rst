Hyperparameters
=============================


General parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

======================  ============  =========================================
Argument                  Type                Help
======================  ============  =========================================
--seed                    int          random seed to be used
--use_cuda                bool          Use GPU
======================  ============  =========================================

Data parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

======================  ============  ==================================================================
Argument                  Type                Help
======================  ============  ==================================================================
--dump_dir                str         path where data and models are dumped
--norm                    str         Feature normalization used in training/validation: None, perfilter, global, cosmo, cosmo_quantile
--redshift                str         Host redshift used in training/validation: zpho, zspe or None
--source_data             str         Data source: photometry or salt
--no_overwrite            bool        If True, overwrite preprocessed dir when creating database
--data_fraction           float       Fraction of data to use
--override_source_data    str         Change the source data (use saltfit or photometry)
--sntypes                 dict        SN type mapping (e.g. '{"101":"Ia","120":"IIP"}'). Types in data not listed here are auto-assigned to a ``contaminant`` class.
--target_sntype           str         Class value in --sntypes to use as target 0 for binary classification (default: Ia)
--sntype_var              str         Column name for event types (default: SNTYPE)
--nb_classes              int         Number of classification targets (default: 2)
======================  ============  ==================================================================

Training parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

======================  ============  ==================================================================
Argument                  Type                Help
======================  ============  ==================================================================
--train_rnn               bool         Train RNN model
--monitor_interval        int          Validate every monitor_interval epochs--metrics
======================  ============  ==================================================================


Validation Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

======================  ============  =====================================================
Argument                  Type                Help
======================  ============  =====================================================
--speed                   bool         Run RNN speed classification benchmark
--calibration             bool         Evaluate model calibration
--performance             bool         Get performance metrics + plots
--metrics                 bool         Compute performance metrics
--model_files             bool         Path to model files
--prediction_files        bool         Path to prediction files
--metric_files            bool         Path to metric files
======================  ============  =====================================================


Visualization Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

===============================  ============  ==========================================================
Argument                          Type                Help
===============================  ============  ==========================================================
--explore_lightcurves             bool         Plot a random selection of lightcurves
--plot_lcs                        bool         Plot a random selection of lightcurves  predictions 
--plot_prediction_distribution    bool         Plot lcs and the histogram of probability for each class
===============================  ============  ==========================================================



RNN parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

========================  ============  ==================================================================
Argument                  Type                Help
========================  ============  ==================================================================
--cyclic                  bool          Use cyclic learning rate
--cyclic_phases           list          Cyclic phases
--random_length           bool          Use random length sequences for training
--random_redshift         bool          If True, randomly set the spectroscopic redshift
--weight_decay            float         L2 decay on weights (for variational RNN)
--layer_type              str           Recurrent layer type. Choose lstm,gru,rnn
--model                   str           Recurrent model type. Choose vanilla,variational,bayesian
--learning_rate           float         Learning rate
--nb_classes              int           Number of classification targets
--nb_epoch                int           Number of epoch
--batch_size              int           Batch size
--hidden_dim              int           Hidden layer dimension
--num_layers              int           Number of recurrent layers
--dropout                 float         Dropout value
--bidirectional           bool          Use bidirectional models
--rnn_output_option       str           RNN output options. standard or mean
--pi                      float         mixing coefficient for Bayes prior
--log_sigma1              float         Initialization parameter for BayesRNN layers
--log_sigma2              float         Initialization parameter for BayesRNN layers
--rho_scale_lower         float         Initialization parameter for BayesRNN layers
--rho_scale_upper         float         Initialization parameter for BayesRNN layers
--log_sigma1_output       float         Initialization parameter for BayesLinear output layers
--log_sigma2_output       float         Initialization parameter for BayesLinear output layers
--rho_scale_lower_output  float         Initialization parameter for BayesLinear output layers
--rho_scale_upper_output  float         Initialization parameter for BayesLinear output layers
--num_inference_samples   int           Number of samples to use for Bayesian inference
--mean_field_inference    bool          Use mean field inference for bayesian models
========================  ============  ==================================================================
