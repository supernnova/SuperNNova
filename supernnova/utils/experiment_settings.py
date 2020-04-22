import os
import json
import h5py
import itertools
import numpy as np
from pathlib import Path
from collections import OrderedDict
from .data_utils import PLASTICC_FILTERS


class ExperimentSettings:
    """Mother class to control experiment parameters

    This class is responsible for the following

    - Defining paths and model names
    - Choosing the device on which to run computations
    - Specifying all hyperparameters such as model configuration, datasets, features etc

    Args:
        cli_args (argparse.Namespace) command line arguments
    """

    def __init__(self, cli_args):

        # Transfer attributes
        if isinstance(cli_args, dict):
            self.__dict__.update(cli_args)
            self.cli_args = cli_args
        else:
            self.__dict__.update(cli_args.__dict__)
            self.cli_args = cli_args.__dict__

        self.device = "cpu"
        if self.use_cuda:
            self.device = "cuda"

        if self.model == "variational":
            self.weight_decay = self.weight_decay
        else:
            self.weight_decay = 0.0

        # Load simulation and training settings and prepare directories
        if self.no_dump:
            pass
        else:
            self.setup_dir()
            # Set the database file names
            self.set_database_file_names()
        
            self.randomforest_features = self.get_randomforest_features()

            # Set the feature lists
            self.set_feature_lists()

            self.overwrite = not self.no_overwrite

            # filter combination
            list_filters_combination = []
            for i in range(1, len(self.list_filters)+1):
                tmp = [''.join(t) for t in list(
                    itertools.combinations(self.list_filters, i))]
                list_filters_combination = list_filters_combination + tmp
            self.list_filters_combination = list_filters_combination

            self.set_randomforest_model_name()
            self.set_pytorch_model_name()
            # Get the feature normalization dict
            self.load_normalization()

    def get_randomforest_features(self):
        """Specify list of features to be used for RandomForest training
        """

        features = [
            "x1",
            "x1ERR",
            "c",
            "cERR",
            "mB",
            "mBERR",
            "x0",
            "x0ERR",  # 'COV_x1_c', 'COV_x1_x0','COV_c_x0', 'NDOF',
            "FITCHI2",
            "m0obs_r",
            "m0obs_i",
            "m0obs_g",
            "m0obs_z",
            "em0obs_i",
            "em0obs_r",
            "em0obs_g",
            "em0obs_z",
        ]

        if self.redshift == "zpho":
            features += ["HOSTGAL_PHOTOZ", "HOSTGAL_PHOTOZ_ERR"]
        elif self.redshift == "zspe":
            features += ["HOSTGAL_SPECZ", "HOSTGAL_SPECZ_ERR"]

        return features

    def setup_dir(self):
        """Configure directories where data is read from or dumped to
        during the course of an experiment
        """

        for path in [
            f"{self.raw_dir}",
            f"{self.fits_dir}",
            f"{self.dump_dir}/explore",
            f"{self.dump_dir}/stats",
            f"{self.dump_dir}/figures",
            f"{self.dump_dir}/lightcurves",
            f"{self.dump_dir}/latex",
            f"{self.dump_dir}/processed",
            f"{self.dump_dir}/preprocessed",
            f"{self.dump_dir}/models",
        ]:

            setattr(self, Path(path).name + "_dir", path)

            Path(path).mkdir(exist_ok=True, parents=True)

    def set_pytorch_model_name(self):
        """Define the model name for all NN based classifiers
        """
        name = f"{self.model}_S_{self.seed}_CLF_{self.nb_classes}"
        name += f"_R_{self.redshift}"
        name += f"_{self.source_data}_DF_{self.data_fraction}_N_{self.norm}"
        name += f"_{self.layer_type}_{self.hidden_dim}x{self.num_layers}"
        name += f"_{self.dropout}"
        name += f"_{self.batch_size}"
        name += f"_{self.bidirectional}"
        name += f"_{self.rnn_output_option}"
        if "bayesian" in self.model:
            name += (
                f"_Bayes_{self.pi}_{self.log_sigma1}_{self.log_sigma2}"
                f"_{self.rho_scale_lower}_{self.rho_scale_upper}"
                f"_{self.log_sigma1_output}_{self.log_sigma2_output}"
                f"_{self.rho_scale_lower_output}_{self.rho_scale_upper_output}"
            )
        if self.cyclic:
            name += "_C"
        if self.weight_decay > 0:
            name += f"_WD_{self.weight_decay}"

        self.pytorch_model_name = name
        self.rnn_dir = f"{self.models_dir}/{self.pytorch_model_name}"

        # deserializing numpy arrays to save as json
        d_tmp = {}
        for k, v in self.__dict__.items():
            if isinstance(v,np.ndarray):
                v = v.tolist()
            d_tmp[k] = v

        if self.train_rnn:
            os.makedirs(self.rnn_dir, exist_ok=True)
            # Dump the command line arguments (for model restoration)
            with open(Path(self.rnn_dir) / "cli_args.json", "w") as f:
                json.dump(d_tmp, f, indent=4, sort_keys=True)

    def set_randomforest_model_name(self):
        """Define the model name for all RandomForest based classifiers
        """

        name = f"randomforest_S_{self.seed}_CLF_{self.nb_classes}"
        name += f"_R_{self.redshift}"
        name += f"_{self.source_data}_DF_{self.data_fraction}_N_{self.norm}"

        self.randomforest_model_name = name
        self.rf_dir = f"{self.models_dir}/{self.randomforest_model_name}"

        if self.train_rf:
            os.makedirs(self.rf_dir, exist_ok=True)
            # Dump the command line arguments (for model restoration)
            with open(Path(self.rf_dir) / "cli_args.json", "w") as f:
                json.dump(self.cli_args, f, indent=4, sort_keys=True)

        return name

    def check_data_exists(self):
        """Utility to check the database has been built
        """

        database_file = f"{self.processed_dir}/database.h5"
        assert os.path.isfile(database_file)

    def set_feature_lists(self):
        """Utility to define the features used to train NN=based models
        """

        self.training_features_to_normalize = [
            f"FLUXCAL_{f}" for f in self.list_filters
        ]
        self.training_features_to_normalize += [
            f"FLUXCALERR_{f}" for f in self.list_filters
        ]
        self.training_features_to_normalize += ["delta_time"]

        if not self.data:
            # If the database has been created, add the list of all features
            with h5py.File(self.hdf5_file_name, "r") as hf:
                self.all_features = hf["features"][:].astype(str)

                self.non_redshift_features = [
                    f for f in self.all_features if "HOSTGAL" not in f
                ]

                # Optionally add redshift
                self.redshift_features = []
                if self.redshift == "zpho":
                    self.redshift_features = [
                        f for f in self.all_features if "HOSTGAL_PHOTOZ" in f
                    ]
                elif self.redshift == "zspe":
                    self.redshift_features = [
                        f for f in self.all_features if "HOSTGAL_SPECZ" in f
                    ]

                self.training_features = (
                    self.non_redshift_features + self.redshift_features
                )

    def set_database_file_names(self):
        """Create a unique database name based on the dataset required
        by the settings
        """

        out_file = f"{self.processed_dir}/database"
        self.pickle_file_name = out_file + ".pickle"
        self.hdf5_file_name = out_file + ".h5"

    def load_normalization(self):
        """Create an array holding the data-normalization parameters
        used to normalize certain features in the NN-based classification
        pipeline
        """

        if not self.data:

            self.idx_features = [
                i
                for (i, f) in enumerate(self.all_features)
                if f in self.training_features
            ]

            self.idx_specz = [
                i
                for (i, f) in enumerate(self.training_features)
                if "HOSTGAL_SPECZ" in f
            ]

            self.idx_flux = [
                i for (i, f) in enumerate(self.training_features) if "FLUXCAL_" in f
            ]

            self.idx_fluxerr = [
                i for (i, f) in enumerate(self.training_features) if "FLUXCALERR_" in f
            ]

            self.idx_delta_time = [
                i for (i, f) in enumerate(self.training_features) if "delta_time" in f
            ]

            self.idx_features_to_normalize = [
                i
                for (i, f) in enumerate(self.all_features)
                if f in self.training_features_to_normalize
            ]

            self.d_feat_to_idx = {f: i for i,
                                  f in enumerate(self.all_features)}

            list_norm = []

            with h5py.File(self.hdf5_file_name, "r") as hf:

                for f in self.training_features_to_normalize:

                    if self.norm == "perfilter":

                        minv = np.array(hf[f"normalizations/{f}/min"])
                        meanv = np.array(hf[f"normalizations/{f}/mean"])
                        stdv = np.array(hf[f"normalizations/{f}/std"])

                        list_norm.append([minv, meanv, stdv])

                    else:

                        if "FLUX" in f:
                            prefix = f.split("_")[0]
                            minv = np.array(hf[f"normalizations_global/{prefix}/min"])
                            meanv = np.array(hf[f"normalizations_global/{prefix}/mean"])
                            stdv = np.array(hf[f"normalizations_global/{prefix}/std"])
                        else:
                            minv = np.array(hf[f"normalizations/{f}/min"])
                            meanv = np.array(hf[f"normalizations/{f}/mean"])
                            stdv = np.array(hf[f"normalizations/{f}/std"])

                        list_norm.append([minv, meanv, stdv])

            self.arr_norm = np.array(list_norm)


class PlasticcSettings(object):

    def __init__(self, cli_args):

        # Transfer attributes
        if isinstance(cli_args, dict):
            self.__dict__.update(cli_args)
        else:
            self.__dict__.update(cli_args.__dict__)

        self.dump_dir = os.path.join(self.dump_dir, "plasticc")
        self.nb_classes = 14

        self.device = "cpu"
        if self.use_cuda:
            self.device = "cuda"

        if self.model == "variational":
            self.weight_decay = self.weight_decay
        else:
            self.weight_decay = 0.0

        # Load simulation and training settings and prepare directories
        self.setup_dir()

        self.pytorch_model_name = self.get_pytorch_model_name()

        # Set the filters used in the study
        self.list_filters = PLASTICC_FILTERS

        # Set the database file names
        self.set_database_file_names()

        # Set the feature lists
        self.set_feature_lists()

        # Get the feature normalization dict
        self.load_normalization()

        # TODO
        # Set SN types for multiclass classification

        self.overwrite = not self.no_overwrite

    def check_data_exists(self):

        database_file = f"{self.processed_dir}/database.h5"
        assert os.path.isfile(database_file)

    def setup_dir(self):

        # Check raw data exists in the dump_dir
        assert os.path.exists(self.raw_dir)

        for path in [
            f"{self.dump_dir}/explore",
            f"{self.dump_dir}/stats",
            f"{self.dump_dir}/figures",
            f"{self.dump_dir}/latex",
            f"{self.dump_dir}/processed",
            f"{self.dump_dir}/preprocessed",
            f"{self.dump_dir}/models",
        ]:

            setattr(self, Path(path).name, path)

            Path(path).mkdir(exist_ok=True, parents=True)

    def get_pytorch_model_name(self):

        name = f"{self.model}_S_{self.seed}_CLF_{self.nb_classes}"
        name += f"_R_{self.redshift}"
        name += f"_{self.source_data}_DF_{self.data_fraction}_N_{self.norm}"
        name += f"_{self.layer_type}_{self.hidden_dim}x{self.num_layers}"
        name += f"_{self.dropout}"
        name += f"_{self.batch_size}"
        name += f"_{self.bidirectional}"
        name += f"_{self.rnn_output_option}"
        if "bayesian" in self.model:
            name += (
                f"_Bayes_{self.pi}_{self.log_sigma1}_{self.log_sigma2}_{self.rho_scale_lower}_{self.rho_scale_upper}"
                f"_{self.log_sigma1_output}_{self.log_sigma2_output}_{self.rho_scale_lower_output}_{self.rho_scale_upper_output}"
            )
        if self.cyclic:
            name += "_C"
        if self.weight_decay > 0:
            name += f"_WD_{self.weight_decay}"

        if self.train_plasticc:
            os.makedirs(os.path.join(self.models_dir, name), exist_ok=True)

        return name

    def set_feature_lists(self):

        self.training_features_to_normalize = [
            f"FLUXCAL_{f}" for f in self.list_filters
        ]
        self.training_features_to_normalize += [
            f"FLUXCALERR_{f}" for f in self.list_filters
        ]
        self.training_features_to_normalize += ["delta_time"]

        if self.train_plasticc or self.predict_plasticc:
            # If the database has been created, add the list of all features
            with h5py.File(self.hdf5_file_name, "r") as hf:
                self.all_features = hf["features"][:].astype(str)

                # Optionally add redshift
                self.redshift_features = []
                if self.redshift == "zpho":
                    # use only photometric redshift features
                    self.redshift_features = [
                        f for f in self.all_features if "HOSTGAL_PHOTOZ" in f
                    ]
                elif self.redshift == "zspe":
                    # Use all redshift features
                    self.redshift_features = [
                        f for f in self.all_features if "HOST" in f
                    ]

                self.non_redshift_features = [
                    f for f in self.all_features if "HOST" not in f
                ]

                self.training_features = (
                    self.non_redshift_features + self.redshift_features
                )
                self.training_features = [
                    f for f in self.all_features if f in self.training_features
                ]

    def set_database_file_names(self):
        out_file = f"{self.processed_dir}/database"
        self.pickle_file_name = out_file + ".pickle"
        self.hdf5_file_name = out_file + ".h5"

    def load_normalization(self):

        if self.train_plasticc or self.predict_plasticc:

            self.idx_features = [
                i
                for (i, f) in enumerate(self.all_features)
                if f in self.training_features
            ]

            self.idx_specz = [
                i
                for (i, f) in enumerate(self.training_features)
                if "HOSTGAL_SPECZ" in f
            ]

            self.idx_features_to_normalize = [
                i
                for (i, f) in enumerate(self.all_features)
                if f in self.training_features_to_normalize
            ]

            self.d_feat_to_idx = {f: i for i,
                                  f in enumerate(self.all_features)}

            list_norm = []

            with h5py.File(self.hdf5_file_name, "r") as hf:

                for f in self.training_features_to_normalize:

                    if self.norm == "perfilter":

                        minv = np.array(hf[f"normalizations/{f}/min"])
                        meanv = np.array(hf[f"normalizations/{f}/mean"])
                        stdv = np.array(hf[f"normalizations/{f}/std"])

                        list_norm.append([minv, meanv, stdv])

                    else:

                        if "FLUX" in f:
                            prefix = f.split("_")[0]
                            minv = np.array(hf[f"normalizations_global/{prefix}/min"])
                            meanv = np.array(hf[f"normalizations_global/{prefix}/mean"])
                            stdv = np.array(hf[f"normalizations_global/{prefix}/std"])
                        else:
                            minv = np.array(hf[f"normalizations/{f}/min"])
                            meanv = np.array(hf[f"normalizations/{f}/mean"])
                            stdv = np.array(hf[f"normalizations/{f}/std"])

                        list_norm.append([minv, meanv, stdv])

            self.arr_norm = np.array(list_norm)
