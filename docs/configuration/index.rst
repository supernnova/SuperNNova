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
--data                    bool        if True, launch data creation
--dump_dir                str         path where data and models are dumped
--redshift                str         Host redshift used in classification: zpho, zspe or None
--norm                    str         Feature normalization: None, perfilter, global
--source_data             str         Data source: photometry or salt
--no_overwrite            bool        If True, overwrite preprocessed dir when creating database
--data_fraction           float       Fraction of data to use
======================  ============  ==================================================================

Training parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

======================  ============  ==================================================================
Argument                  Type                Help
======================  ============  ==================================================================
--train_rnn               bool         Train RNN model
--train_rf                bool         Train RandomForest model
======================  ============  ==================================================================


Validation Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

======================  ============  =====================================================
Argument                  Type                Help
======================  ============  =====================================================
--validate_rnn            bool         Validate RNN model
--validate_rf             bool         Validate RandomForest model
--speed                   bool         Run RNN speed classification benchmark
--calibration             bool         Evaluate model calibration
--performance             bool         Get performance metrics + plots
--metrics                 bool         Compute performance metrics
--science_plots           bool         Plots of scientific interest
======================  ============  =====================================================


Visualization Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

======================  ============  =====================================================
Argument                  Type                Help
======================  ============  =====================================================
--explore_lightcurves     bool         Plot a random selection of lightcurves
--plot_lcs                bool         Plot a random selection of lightcurves  predictions
======================  ============  =====================================================



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
--KLfactor                float         Factor for KL regularisation (BBB RNN)
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


Random Forest parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

======================  ============  ==================================================================
Argument                  Type                Help
======================  ============  ==================================================================
--bootstrap              bool         Activate bootstrap when building trees
--min_samples_leaf       int          Minimum samples required to be a leaf node
--n_estimators           int          Number of trees
--min_samples_split      int          Min samples to create split
--criterion              str          Tree splitting criterion
--max_features           int          Max features per tree
--max_depth              int          Max tree depth
======================  ============  ==================================================================