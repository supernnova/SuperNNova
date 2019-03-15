import numpy as np
from pathlib import Path
from ..utils import data_utils
from ..utils import training_utils
from ..utils import logging_utils


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

    prediction_file = (
        f"{dump_dir}/PRED_{settings.randomforest_model_name}.pickle"
    )

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
