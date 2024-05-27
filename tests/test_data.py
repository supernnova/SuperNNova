import os
import shutil
from .test_utils import call_cmd
import pytest

dir_data_path = os.path.dirname(os.path.realpath(__file__)) + "/dump_data/"


@pytest.mark.parametrize(
    "option",
    [
        "",
        "--list_filters g r",
        "--testing_ids tests/raw_csv/DES_HEAD.csv",
        '--sntypes \'{"101":"Ia"}\' ',
        "--additional_train_var MWEBV",
    ],
)
def test_dataset_making(option):
    shutil.rmtree(dir_data_path, ignore_errors=True)

    cmd = "snn make_data --dump_dir tests/dump_data --raw_dir tests/raw " + option
    call_cmd(cmd)

    # check whether database has been generated
    assert os.path.exists(dir_data_path + "processed/database.h5") is True
    assert os.path.exists(dir_data_path + "processed/SNID.pickle") is True
    assert os.path.exists(dir_data_path + "processed/hostspe_SNID.pickle") is True
    shutil.rmtree(dir_data_path)


def test_explore_lightcurves():
    shutil.rmtree(dir_data_path, ignore_errors=True)
    cmd = "snn make_data --dump_dir tests/dump_data --raw_dir tests/raw --explore_lightcurves --debug"
    call_cmd(cmd)

    # check whether database has been generated
    assert os.path.exists(dir_data_path + "processed/database.h5") is True
    assert os.path.exists(dir_data_path + "processed/SNID.pickle") is True
    assert os.path.exists(dir_data_path + "processed/hostspe_SNID.pickle") is True

    # check whether plots have been generated
    assert os.path.exists(dir_data_path + "explore/sample_lightcurves.png") is True
    assert (
        os.path.exists(dir_data_path + "explore/sample_lightcurves_from_hdf5.png")
        is True
    )
    shutil.rmtree(dir_data_path)
