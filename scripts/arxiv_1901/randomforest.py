import yaml
import argparse
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from supernnova.utils import data_utils
from supernnova.utils import training_utils
from supernnova.utils import logging_utils
from supernnova.validation import metrics

from constants import SNTYPES


def train(config):
    """Train random forest models

    - Features are stored in a .FITRES file found in settings.processed_dir.
    - Carry out a train/val split based on predefined splits
    - Compute evaluation metrics on train and validation set
    - Save trained models to settings.models_dir

    Args:
        settings: (ExperimentSettings) custom class to hold hyperparameters
    """

    clf = RandomForestClassifier(
        bootstrap=config["bootstrap"],
        min_samples_leaf=config["min_samples_leaf"],
        n_estimators=config["n_estimators"],
        min_samples_split=config["min_samples_split"],
        criterion=config["criterion"],
        max_features=config["max_features"],
        max_depth=config["max_depth"],
        random_state=config["seed"],
        n_jobs=-1,
    )

    ###################
    # Data
    ###################

    # load data
    df_data = data_utils.load_fitfile(config["fitopt_file"], SNTYPES)
    SNID = df_data["SNID"].values

    # Load training and validation SNID to make our train/val split
    sn_df = data_utils.load_HDF5_SNinfo(config["processed_dir"])

    class_map = {}
    for key, value in SNTYPES.items():
        class_map[int(key)] = 0 if value == "Ia" else 1

    sn_df["target"] = sn_df["SNTYPE"].map(class_map)

    # Balance classes
    target = sn_df["target"].values
    idxs_0 = np.where(target == 0)[0]
    idxs_1 = np.where(target == 1)[0]

    n_samples = min(len(idxs_0), len(idxs_1))
    idxs_0 = np.random.choice(idxs_0, size=n_samples, replace=False)
    idxs_1 = np.random.choice(idxs_1, size=n_samples, replace=False)

    idxs_keep = np.concatenate([idxs_0, idxs_1])
    np.random.shuffle(idxs_keep)
    sn_df = sn_df.iloc[idxs_keep].reset_index(drop=True)

    n_0 = sn_df[sn_df.target == 0].shape[0]
    n_1 = sn_df[sn_df.target == 1].shape[0]
    n = sn_df.shape[0]

    print(f"{n_0} ({100 * n_0 / n:.2f} %) class 0 samples after balancing")
    print(f"{n_1} ({100 * n_1 / n:.2f} %) class 1 samples after balancing")

    # Train val split
    n_train = int(0.8 * n)
    df_train = sn_df[:n_train]
    df_val = sn_df[n_train:]

    features = config["features"]

    # Avoid duplicate columns
    cols = [c for c in df_data.columns if c not in df_train.columns] + ["SNID"]
    df_data = df_data[cols]

    df_train = df_train.merge(df_data, on="SNID", how="left")[features + ["target"]]
    df_val = df_val.merge(df_data, on="SNID", how="left")[features + ["target"]]

    ###################
    # Training
    ###################

    # Prepare features and target
    X_train = df_train[features].values
    y_train = df_train["target"].values

    X_val = df_val[features].values
    y_val = df_val["target"].values

    logging_utils.print_green("Features", ",".join(features))

    # Fit  and evaluate model
    clf = training_utils.train_and_evaluate_randomforest_model(
        clf, X_train, y_train, X_val, y_val
    )
    # save the model to disk
    save_file = (Path(config["dump_dir"]) / f"{config['model_name']}.pickle").as_posix()
    Path(save_file).parent.mkdir(exist_ok=True, parents=True)
    training_utils.save_randomforest_model(save_file, clf)


def get_predictions(settings, model_file=None):
    """Test random forest models on independent test set

    Features are stored in a .FITRES file found in data_dir
    Use predefined splits to select test set
    Save predicted target and probabilities to preds_dir

    Args:
        settings (ExperimentSettings): custom class to hold hyperparameters
        model_file (str): path to saved randomforest model
    """

    assert settings.source_data == "saltfit", logging_utils.str_to_redstr(
        "Only salfit is a valid data source for random forest"
    )
    assert settings.nb_classes == 2, logging_utils.str_to_redstr(
        "Binary classification is the only task allowed for RandomForest"
    )

    if model_file is None:
        dump_dir = f"{settings.models_dir}/{settings.randomforest_model_name}"
        model_file = f"{dump_dir}/{settings.randomforest_model_name}.pickle"
    else:
        dump_dir = Path(model_file).parent

    if settings.override_source_data is not None:
        settings.source_data = settings.override_source_data
        settings.set_pytorch_model_name()

    prediction_file = f"{dump_dir}/PRED_{settings.randomforest_model_name}.pickle"

    # load data
    df_test = data_utils.load_fitfile(settings)

    # Load the dataframe containing  the list of the independent test SNID
    sn_df = data_utils.load_HDF5_SNinfo(settings)
    df_SNID = sn_df[sn_df["dataset_saltfit_2classes"] == 2][
        ["SNID", "PEAKMJD", "PEAKMJDNORM", "SIM_REDSHIFT_CMB", "SNTYPE"]
    ]

    # Make sure SNID is of type int
    df_SNID["SNID"] = df_SNID["SNID"].astype(int)
    df_test["SNID"] = df_test["SNID"].astype(int)

    # Select only IDs that are in df_SNID and df_test
    df_test = df_test.merge(df_SNID, on="SNID")

    # Add redshift features if required by settings
    df_test = data_utils.add_redshift_features(settings, df_test)

    # Load randomforest model
    clf = training_utils.load_randomforest_model(settings, model_file=model_file)
    # Make predictions and save to dataframe
    X = df_test[settings.randomforest_features].values
    y_pred_proba = clf.predict_proba(X)
    y_pred_class = np.argmax(y_pred_proba, axis=1)

    df_test["all_class0"] = y_pred_proba[:, 0]
    df_test["all_class1"] = y_pred_proba[:, 1]
    df_test["predicted_target"] = y_pred_class
    df_test["target"] = df_test["target_2classes"]

    # Cache results for future re-use
    list_features_save = [
        "SNID",
        "all_class0",
        "all_class1",
        "predicted_target",
        "target",
    ]
    df_test[list_features_save].to_pickle(prediction_file)

    return prediction_file


def main(config_path):

    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    # setting random seeds
    np.random.seed(config["seed"])

    train(config)
    import ipdb

    ipdb.set_trace()
    # Obtain predictions
    get_predictions(config)
    # Compute metrics
    metrics.get_metrics_singlemodel(config, model_type="rf")

    logging_utils.print_blue("Finished rf training, validating and testing")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dataset building")
    parser.add_argument("config_path", help="Path to yml config gile")

    args = parser.parse_args()

    main(args.config_path)
