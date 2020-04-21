# INPUT
# Beware, there may be missing filters

# option 1: Input format = lists
mjd = [57433.4816, 57436.4815]
fluxes = [2.0, 3]
fluxerrs = [0.1, 0.2]
passbands = ["g", "r"]
SNID = "1"
redshift_zspe = 0.12
redshift_zpho = 0.1
redshift = 0.12
# redshift can be given either as zpho/zspe of global one and use zpho model

# option 2: small pandas dataframe
df = pd.DataFrame()
df["mjd"] = [57433.4816, 57436.4815, 33444, 33454]
df["fluxes"] = [2.0, 3, 200, 300]
df["fluxerrs"] = [0.1, 0.2, 0.1, 0.2]
df["passbands"] = ["g", "r", "g", "r"]
df["SNID"] = ["1", "1", "2", "2"]
df["redshift_zspe"] = [0.12, 0.12, 0.5, 0.5]
df["redshift_zpho"] = [0.1, 0.1, 0.5, 0.5]
df["redshift"] = [0.12, 0.12, 0.5, 0.5]

# VALIDATE
# Important: want classification output directly here, not in a file.

# USAGE
import supernnova.conf as conf
from supernnova.data import ontheflydata
from supernnova.validation import validate_rnn_onthefly

# read data
ontheflydata(df) or ontheflydata(mjd, fluxes, fluxerrs, redshift)

# get config args
args = conf.get_args()
args.validate_rnn = False  # conf: validate rnn
args.model_files = "model_file"  # conf: model file to load
settings = conf.get_settings(args)  # conf: set settings
preds = validate_rnn_onthefly.get_predictions(settings)  # classify test set

# output format, list with predictions
[0.5, 0.6]


# to check predictions you can use early_predictions and save lcs as df
arr_flux = []
arr_fluxerr = []
arr_flt = []
arr_MJD = []
for flt in ["g", "r", "i", "z"]:
    arr_flux += df_temp[f"FLUXCAL_{flt}"].values.tolist()
    arr_fluxerr += df_temp[f"FLUXCALERR_{flt}"].values.tolist()
    arr_MJD += arr_time.tolist()
    arr_flt += flt * len(df_temp[f"FLUXCAL_{flt}"].values.tolist())
aaa = pd.DataFrame()
aaa["FLUXCAL"] = arr_flux
aaa["FLUXCALERR"] = arr_fluxerr
aaa["FLT"] = arr_flt
aaa["MJD"] = arr_MJD
aaa["SNID"] = np.ones(len(aaa)).astype(int).astype(str)
aaa.to_csv("tmp_cl.csv")
