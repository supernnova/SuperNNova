import os
import h5py
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from natsort import natsorted

import torch
import torch.nn as nn

from ..utils import data_utils as du
from ..utils import training_utils as tu
from ..utils import logging_utils as lu


def load_plasticc_test(hdf5_file_name, settings):
    """Load data from HDF5
    """

    with h5py.File(hdf5_file_name, "r") as hf:

        n_features = hf["data"].attrs["n_features"]

        arr_data = hf["data"][:]
        arr_SNID = hf["SNID"][:]
        n_samples = arr_data.shape[0]

        # Create a dummy target
        arr_target = np.zeros(n_samples)

        return tu.fill_data_list(
            np.arange(n_samples),
            arr_data,
            arr_target,
            arr_SNID,
            settings,
            n_features,
            "Loading Test Set",
            test=False,
        )


def get_predictions(settings):

    settings.random_length = False
    settings.random_redshift = False

    if "vanilla" in settings.pytorch_model_name:
        settings.num_inference_samples = 1

    # Load RNN model
    rnn = tu.get_model(settings, len(settings.training_features))

    dump_dir = f"{settings.models_dir}/{settings.pytorch_model_name}"
    model_file = f"{dump_dir}/{settings.pytorch_model_name}.pt"

    rnn_state = torch.load(model_file, map_location=lambda storage, loc: storage)
    rnn.load_state_dict(rnn_state)
    rnn.to(settings.device)
    rnn.eval()

    # Load the data
    list_test_hdf5 = glob.glob(os.path.join(settings.processed_dir, "*test_set_*.h5"))
    # Sort the hdf5 files
    list_test_hdf5 = natsorted(list_test_hdf5)

    #####################################
    # Loop over test databases and predict
    #####################################

    list_df = []

    for hdf5_file_name in tqdm(list_test_hdf5, ncols=100):

        # Load test data
        list_data_test = load_plasticc_test(hdf5_file_name, settings)
        num_samples = len(list_data_test)

        # Split into batches
        batch_size = min(2048, num_samples)
        num_batches = num_samples / batch_size
        list_batches = np.array_split(np.arange(num_samples), num_batches)
        for index, batch_idxs in enumerate(list_batches):

            # Get the object ID (or SNID)
            arr_SNID = (
                np.array([list_data_test[i][2] for i in batch_idxs])
                .astype(int)
                .reshape(-1, 1)
            )

            with torch.no_grad():

                packed, _, target_tensor, idxs_rev_sort = tu.get_data_batch(
                    list_data_test, batch_idxs, settings
                )

                # Predict
                if "vanilla" in settings.pytorch_model_name:
                    out = rnn.forward(packed)
                    arr_preds = nn.functional.softmax(out, dim=-1).data.cpu().numpy()

                elif settings.mean_field_inference is True:
                    out = rnn.forward(packed, mean_field_inference=True)
                    arr_preds = nn.functional.softmax(out, dim=-1).data.cpu().numpy()

                else:

                    # Unpack the array, reshape it for multiple predictions
                    unpacked_tensor, lengths = nn.utils.rnn.pad_packed_sequence(packed)

                    L, B, D = unpacked_tensor.shape

                    # Copy the data num_inference_sample times for predictions over a new dimension
                    unpacked_tensor = (
                        unpacked_tensor.view(L, B, 1, D)
                        .expand(L, B, settings.num_inference_samples, D)
                        .clone()
                    )
                    lengths = (
                        lengths.view(B, 1)
                        .expand(B, settings.num_inference_samples)
                        .clone()
                    )

                    # Then reflatten the array to pass them to the model
                    unpacked_tensor = unpacked_tensor.view(
                        L, B * settings.num_inference_samples, D
                    )
                    lengths = lengths.view(B * settings.num_inference_samples)

                    # Pack the data
                    packed = nn.utils.rnn.pack_padded_sequence(unpacked_tensor, lengths)

                    out = rnn.forward(packed)

                    # Obtain predictions, reshape them from (B * settings.num_inference_samples , nb_classes)
                    # to (B, settings.num_inference_samples, nb_classes) and take the mean
                    arr_preds = (
                        nn.functional.softmax(out, dim=-1)
                        .data.cpu()
                        .numpy()
                        .reshape((B, settings.num_inference_samples, -1))
                        .mean(axis=1)
                    )

                # Revert sorting
                arr_preds = arr_preds[idxs_rev_sort]

                df_pred = pd.DataFrame(
                    data=arr_preds,
                    columns=[f"class_{i}" for i in range(arr_preds.shape[-1])],
                )
                df_pred["object_id"] = arr_SNID

            list_df.append(df_pred)

    # Concat all predictions
    df = pd.concat(list_df)
    # Change header name
    df = df.rename(
        columns={
            f"class_{value}": f"class_{key}"
            for (key, value) in du.DICT_PLASTICC_CLASS.items()
        }
    )
    df = df.sort_values("object_id")

    # Add class 99 column
    df["class_99"] = np.zeros(len(df))

    # Reorder columns
    id_column = ["object_id"]
    class_columns = [f"class_{key}" for key in du.DICT_PLASTICC_CLASS.keys()]
    df[class_columns] = df[class_columns].astype(np.float16)
    columns = id_column + class_columns
    df = df[columns]

    # Rework class 99
    arr_preds = df[[c for c in class_columns if c != "class_99"]].values

    # max_pred = arr_preds.max(axis=1)
    # arr_99 = df["class_99"].values
    # # Heuristic: when the model is confident in its predictions, leave arr_99 to 0
    # arr_99[max_pred >= 0.8] = 0
    # # Otherwise, assign it to 1 - max_pred
    # arr_99[max_pred < 0.8] = (1 - max_pred)[max_pred < 0.8]
    # df["class_99"] = arr_99

    # Olivier's code
    preds_99 = np.ones(arr_preds.shape[0])
    for i in range(arr_preds.shape[1]):
        preds_99 *= 1 - arr_preds[:, i]

    df["class_99"] = 0.14 * preds_99 / np.mean(preds_99)

    # Renormalize proba
    arr = df[class_columns].values
    arr = arr / arr.sum(1, keepdims=True)
    df[class_columns] = arr

    # Load the metadata test set and check we do have all predictions
    df_meta = pd.read_csv(os.path.join(settings.raw_dir, "test_set_metadata.csv"))

    sorted_id = np.sort(df.object_id.unique())
    sorted_id_from_meta = np.sort(df_meta.object_id.unique())

    # Check we do have all the snid in the prediction set
    assert np.all(sorted_id == sorted_id_from_meta)

    # Save predictions
    prediction_file = f"{dump_dir}/rnn_predictions_{settings.pytorch_model_name}.csv"
    df.to_csv(prediction_file, index=False)
