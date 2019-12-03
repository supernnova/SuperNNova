import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch


class HDF5Dataset:
    def __init__(
        self,
        hdf5_file,
        metadata_features,
        sntypes,
        nb_classes=2,
        data_fraction=1.0,
        SNID_train=None,
        SNID_val=None,
        SNID_test=None,
        load_all=False,
    ):
        super().__init__()

        self.hdf5_file = hdf5_file

        # Load dataset information
        hf = h5py.File(hdf5_file, "r")
        self.arr_SNID = hf["SNID"][:]
        self.arr_SNTYPE = hf["SNTYPE"][:]
        arr_meta = hf["metadata"][:]
        columns = hf["metadata"].attrs["columns"]
        df_meta = pd.DataFrame(arr_meta, columns=columns)
        df_meta["SNID"] = self.arr_SNID
        df_meta["SNTYPE"] = self.arr_SNTYPE
        self.list_features = hf["data"].attrs["columns"].tolist()
        hf.close()

        # Prepare metadata features array
        self.arr_meta = df_meta[metadata_features].values if metadata_features else None

        # Create target
        class_map = {}
        if nb_classes == 2:
            for key, value in sntypes.items():
                class_map[int(key)] = 0 if value == "Ia" else 1
        else:
            for i, key in enumerate(sntypes):
                class_map[int(key)] = i
        df_meta["target"] = df_meta["SNTYPE"].map(class_map)

        self.arr_target = df_meta["target"].values

        if load_all:
            self.splits = {"all": np.arange(self.arr_target.shape[0])}

        else:
            # Subsample with data fraction
            n_samples = int(data_fraction * len(df_meta))
            idxs = np.random.choice(len(df_meta), n_samples, replace=False)
            df_meta = df_meta.iloc[idxs].reset_index(drop=True)

            # Pandas magic to downample each class down to lowest cardinality class
            df_meta = df_meta.groupby("target")
            df_meta = (
                df_meta.apply(lambda x: x.sample(df_meta.size().min()))
                .reset_index(drop=True)
                .sample(frac=1)
            ).reset_index(drop=True)

            n_samples = len(df_meta)

            for t in range(nb_classes):
                n = len(df_meta[df_meta.target == t])
                print(
                    f"{n} ({100 * n / n_samples:.2f} %) class {t} samples after balancing"
                )

            self.SNID_train = SNID_train
            self.SNID_val = SNID_val
            self.SNID_test = SNID_test

            # 80/10/10 Train/val/test split
            n_train = int(0.8 * n)
            n_val = int(0.9 * n)
            if self.SNID_train is None:
                self.SNID_train = df_meta["SNID"].values[:n_train]
            if self.SNID_val is None:
                self.SNID_val = df_meta["SNID"].values[n_train:n_val]
            if self.SNID_test is None:
                self.SNID_test = df_meta["SNID"].values[n_val:]

            train_indices = np.where(np.in1d(self.arr_SNID, self.SNID_train))[0]
            val_indices = np.where(np.in1d(self.arr_SNID, self.SNID_val))[0]
            test_indices = np.where(np.in1d(self.arr_SNID, self.SNID_test))[0]

            # Shuffle for good measure
            np.random.shuffle(train_indices)
            np.random.shuffle(val_indices)
            np.random.shuffle(test_indices)

            self.splits = {
                "train": train_indices,
                "val": val_indices,
                "test": test_indices,
            }

    def __len__(self):
        if "all" in self.splits:
            return len(self.split["all"])
        else:
            return len(self.splits["train"])

    def create_iterator(self, split, batch_size, device, tqdm_desc=None):

        idxs = self.splits[split]
        np.random.shuffle(idxs)

        # Create a list of batches
        list_idxs = [
            sorted(idxs[i : i + batch_size].tolist())
            for i in range(0, len(idxs), batch_size)
        ]

        n_features = len(self.list_features)

        flux_features_idxs = [
            i for i in range(n_features) if "FLUXCAL_" in self.list_features[i]
        ]
        fluxerr_features_idxs = [
            i for i in range(n_features) if "FLUXCALERR_" in self.list_features[i]
        ]
        time_idxs = self.list_features.index("delta_time")
        flt_idxs = self.list_features.index("FLT")

        arr_target = self.arr_target
        arr_meta = self.arr_meta
        arr_SNID = self.arr_SNID
        has_meta = arr_meta is not None

        Dflux = len(flux_features_idxs)
        Dfluxerr = len(fluxerr_features_idxs)

        with h5py.File(self.hdf5_file, "r") as hf:

            arr_data = hf["data"]

            iterator = (
                tqdm(
                    list_idxs,
                    desc=tqdm_desc,
                    ncols=100,
                    bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} {rate_fmt}{postfix}",
                )
                if tqdm_desc is not None
                else list_idxs
            )

            for idxs in iterator:

                tmp_X = arr_data[idxs]
                X_SNID = arr_SNID[idxs]
                X_target = arr_target[idxs]
                X_meta = arr_meta[idxs] if has_meta else None

                list_lengths = [X.shape[0] // n_features for X in tmp_X]
                B = len(idxs)
                L = max(list_lengths)

                X_flux = np.zeros((B, L, Dflux), dtype=np.float32)
                X_fluxerr = np.zeros((B, L, Dfluxerr), dtype=np.float32)
                X_time = np.zeros((B, L, 1), dtype=np.float32)
                X_flt = np.zeros((B, L), dtype=np.int64)

                for i in range(B):

                    X = tmp_X[i].reshape(-1, n_features)
                    length = list_lengths[i]

                    X_flux[i, :length, :] = X[:length, flux_features_idxs]
                    X_fluxerr[i, :length, :] = X[:length, fluxerr_features_idxs]
                    X_time[i, :length, 0] = X[:length, time_idxs]
                    X_flt[i, :length] = X[:length, flt_idxs]

                arr_lengths = np.array(list_lengths)
                X_mask = (
                    arr_lengths.reshape(-1, 1) > np.arange(L).reshape(1, -1)
                ).astype(np.bool)

                out = {
                    "X_flux": torch.from_numpy(X_flux).to(device),
                    "X_fluxerr": torch.from_numpy(X_fluxerr).to(device),
                    "X_time": torch.from_numpy(X_time).to(device),
                    "X_flt": torch.from_numpy(X_flt).to(device),
                    "X_target": torch.from_numpy(X_target).to(device),
                    "X_mask": torch.from_numpy(X_mask).to(device),
                    "X_SNID": X_SNID,
                }

                if has_meta:
                    out["X_meta"] = torch.from_numpy(X_meta).to(device)

                yield out
