import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ..utils import data_utils as du
from ..utils import training_utils as tu
from ..utils import logging_utils


def train(settings):
    """Train random forest models

    - Features are stored in a .FITRES file found in settings.processed_dir.
    - Carry out a train/val split based on predefined splits
    - Compute evaluation metrics on train and validation set
    - Save trained models to settings.models_dir

    Args:
        settings: (ExperimentSettings) custom class to hold hyperparameters
    """

    assert settings.source_data == "saltfit", logging_utils.str_to_redstr(
        "Only salfit is a valid data source for random forest"
    )
    assert settings.nb_classes == 2, logging_utils.str_to_redstr(
        "Binary classification is the only task allowed for RandomForest"
    )

    clf = RandomForestClassifier(
        bootstrap=settings.bootstrap,
        min_samples_leaf=settings.min_samples_leaf,
        n_estimators=settings.n_estimators,
        min_samples_split=settings.min_samples_split,
        criterion=settings.criterion,
        max_features=settings.max_features,
        max_depth=settings.max_depth,
        random_state=settings.seed,
        n_jobs=-1,
    )

    ###################
    # Data
    ###################

    # load data
    df_data = du.load_fitfile(settings)
    SNID = df_data["SNID"].values

    # Load training and validation SNID to make our train/val split
    sn_df = du.load_HDF5_SNinfo(settings)
    df_train = sn_df[sn_df["dataset_photometry_2classes"] == 0]
    n_train = len(df_train)
    df_train = df_train[: int(settings.data_fraction * n_train)]

    df_val = sn_df[sn_df["dataset_photometry_2classes"] == 1]
    n_val = len(df_val)
    df_val = df_val[: int(settings.data_fraction * n_val)]

    # Compute train/val split index
    idx_train = np.where(np.in1d(SNID, df_train.SNID.values))[0]
    idx_val = np.where(np.in1d(SNID, df_val.SNID.values))[0]

    # Add redshift features if required by settings
    df_data = du.add_redshift_features(settings, df_data)

    ###################
    # Training
    ###################

    # Prepare features and target
    X = df_data[settings.randomforest_features].values
    y = df_data["target_2classes"].values

    logging_utils.print_green("Features", ",".join(settings.randomforest_features))

    # Apply train/val split
    X_train, X_val = X[idx_train], X[idx_val]
    y_train, y_val = y[idx_train], y[idx_val]

    # Fit  and evaluate model
    clf = tu.train_and_evaluate_randomforest_model(clf, X_train, y_train, X_val, y_val)
    # save the model to disk
    tu.save_randomforest_model(settings, clf)
