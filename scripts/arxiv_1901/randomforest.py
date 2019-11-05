import yaml
import argparse
import numpy as np
import pandas as pd
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
    sn_df = data_utils.load_HDF5_SNinfo(config["processed_dir"])

    # Subsample with data fraction
    n_samples = int(config.get("data_fraction", 1) * sn_df.shape[0])
    idxs = np.random.choice(sn_df.shape[0], n_samples, replace=False)
    sn_df = sn_df.iloc[idxs].reset_index(drop=True)

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

    # 80/10/10 Train/val/test split
    n_train = int(0.8 * n)
    n_val = int(0.9 * n)
    df_train = sn_df[:n_train]
    df_val = sn_df[n_train:n_val]
    df_test = sn_df[n_val:]

    features = config["features"]

    # Avoid duplicate columns
    cols = [c for c in df_data.columns if c not in df_train.columns] + ["SNID"]
    df_data = df_data[cols]

    df_train = df_train.merge(df_data, on="SNID", how="left")[features + ["target"]]
    df_val = df_val.merge(df_data, on="SNID", how="left")[features + ["target"]]
    df_test = df_test.merge(df_data, on="SNID", how="left")[
        features + ["target", "SNID"]
    ]

    ###################
    # Training
    ###################

    # Prepare features and target
    X_train = df_train[features].values
    y_train = df_train["target"].values

    X_val = df_val[features].values
    y_val = df_val["target"].values

    X_test = df_test[features].values

    logging_utils.print_green("Features", ",".join(features))

    # Fit  and evaluate model
    clf = training_utils.train_and_evaluate_randomforest_model(
        clf, X_train, y_train, X_val, y_val
    )
    # save the model to disk
    save_file = (Path(config["dump_dir"]) / f"model.pickle").as_posix()
    Path(save_file).parent.mkdir(exist_ok=True, parents=True)
    training_utils.save_randomforest_model(save_file, clf)

    y_pred_proba = clf.predict_proba(X_test)
    y_pred_class = np.argmax(y_pred_proba, axis=1)

    df_test["all_class0"] = y_pred_proba[:, 0]
    df_test["all_class1"] = y_pred_proba[:, 1]
    df_test["predicted_target"] = y_pred_class

    # Cache results for future re-use
    list_features_save = [
        "SNID",
        "all_class0",
        "all_class1",
        "predicted_target",
        "target",
    ]
    prediction_file = (Path(config["dump_dir"]) / f"PRED.pickle").as_posix()
    df_test[list_features_save].to_pickle(prediction_file)


def get_metrics(config):
    """Launch computation of all evaluation metrics for a given model, specified
    by the settings object or by a model file

    Save a pickled dataframe (we pickle  because we're saving numpy arrays, which
    are not easily savable with the ``to_csv`` method).

    Args:
        settings (ExperimentSettings): custom class to hold hyperparameters
        prediction_file (str): Path to saved predictions. Default: ``None``
        model_type (str): Choose ``rnn`` or ``randomforest``

    Returns:
        (pandas.DataFrame) holds the performance metrics for this dataframe
    """

    processed_dir = config["processed_dir"]
    prediction_file = (Path(config["dump_dir"]) / f"PRED.pickle").as_posix()
    metrics_file = (Path(config["dump_dir"]) / f"METRICS.pickle").as_posix()

    df_SNinfo = data_utils.load_HDF5_SNinfo(config["processed_dir"])
    host = pd.read_pickle(f"{processed_dir}/hostspe_SNID.pickle")
    host_zspe_list = host["SNID"].tolist()

    df = pd.read_pickle(prediction_file)
    df = pd.merge(df, df_SNinfo[["SNID", "SNTYPE"]], on="SNID", how="left")

    list_df_metrics = []

    list_df_metrics.append(metrics.get_calibration_metrics_singlemodel(df))
    list_df_metrics.append(
        metrics.get_randomforest_performance_metrics(df, host_zspe_list, SNTYPES)
    )

    df_metrics = pd.concat(list_df_metrics, 1)

    df_metrics["model_name"] = Path(config["dump_dir"]).name
    df_metrics["source_data"] = "saltfit"
    df_metrics.to_pickle(metrics_file)

    logging_utils.print_green("Finished getting metrics ")


def main(config_path):

    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    # setting random seeds
    np.random.seed(config["seed"])

    # Train and sav predictions
    train(config)
    # Compute metrics
    get_metrics(config)

    logging_utils.print_blue("Finished rf training, validating and testing")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dataset building")
    parser.add_argument("config_path", help="Path to yml config gile")

    args = parser.parse_args()

    main(args.config_path)
