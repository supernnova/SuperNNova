import numpy as np
import pandas as pd
from astropy.table import Table

"""Creating a csv dataset

From SNANA FITS files, create csv counterparts to test processing this format
Using only the first two lcs
"""

keep_col_header = [
    "SNID",
    "PEAKMJD",
    "HOSTGAL_PHOTOZ",
    "HOSTGAL_PHOTOZ_ERR",
    "HOSTGAL_SPECZ",
    "HOSTGAL_SPECZ_ERR",
    "SIM_REDSHIFT_CMB",
    "SIM_PEAKMAG_z",
    "SIM_PEAKMAG_g",
    "SIM_PEAKMAG_r",
    "SIM_PEAKMAG_i",
    "SNTYPE",
]
keep_col_phot = ["SNID", "MJD", "FLUXCAL", "FLUXCALERR", "FLT"]

head = Table.read("../tests/raw/DES_Ia-0001_HEAD.FITS", format="fits")
df_head = pd.DataFrame(np.array(head))
df_head = df_head[:2]
df_head[keep_col_header].to_csv("../tests/raw_csv/DES_HEAD.csv")

phot = Table.read("../tests/raw/DES_Ia-0001_PHOT.FITS", format="fits")
df_phot = pd.DataFrame(np.array(phot))
# 2 first lcs only
arr_idx = np.where(phot["MJD"] == -777.0)[0]
df_phot = df_phot[: arr_idx[1]]
# add snid
df_phot["SNID"] = np.append(
    df_head["SNID"][0] * np.ones(arr_idx[0]),
    df_head["SNID"][1] * np.ones(arr_idx[1] - arr_idx[0]),
).astype(int)
# reformat
df_phot["FLT"] = df_phot["FLT"].values.astype(str)
df_phot[keep_col_phot].to_csv("../tests/raw_csv/DES_PHOT.csv")
