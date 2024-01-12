import os
import shutil
import glob
from .test_utils import call_cmd
import pytest

dir_data_path = os.path.dirname(os.path.realpath(__file__)) + "/dump_visual/"


@pytest.fixture(scope="class")
def setup():
    shutil.rmtree(dir_data_path, ignore_errors=True)
    cmd1 = (
        "python run.py --data --dump_dir tests/dump_visual --raw_dir tests/raw --debug"
    )
    call_cmd(cmd1)
    cmd2 = "python run.py --train_rnn --dump_dir tests/dump_visual --nb_epoch=10"
    call_cmd(cmd2)
    cmd3 = "python run.py --validate_rnn --dump_dir tests/dump_visual"
    call_cmd(cmd3)
    yield
    shutil.rmtree(dir_data_path, ignore_errors=True)


class TestVisualization:
    @pytest.fixture(autouse=True, scope="class")
    def setup_class(self, setup):
        pass

    def test_explore_lightcurve(self):
        cmd = "python run.py --dump_dir tests/dump_visual --explore_lightcurves"
        call_cmd(cmd)
        assert os.path.exists(dir_data_path + "explore/sample_lightcurves.png") is True
        assert (
            os.path.exists(dir_data_path + "explore/sample_lightcurves_from_hdf5.png")
            is True
        )

    def test_plot_lcs(self):
        # remove lightcurves generated during --train_rnn
        shutil.rmtree(dir_data_path + "lightcurves", ignore_errors=True)

        model_dir = dir_data_path + "models/"
        model_files = glob.glob(model_dir + "*/*.pt")
        assert len(model_files) == 1

        model_file = model_files[0]
        cmd = (
            "python run.py --dump_dir tests/dump_visual --plot_lcs --model_files  "
            + model_file
        )
        call_cmd(cmd)

        lc_files = glob.glob(dir_data_path + "lightcurves/*/early_prediction/*.png")
        assert len(lc_files) > 0

    def test_plot_file(self):
        # remove lightcurves generated previously
        shutil.rmtree(dir_data_path + "lightcurves", ignore_errors=True)

        model_dir = dir_data_path + "models/"
        model_files = glob.glob(model_dir + "*/*.pt")
        assert len(model_files) == 1

        model_file = model_files[0]
        # the csv file not exist; it will trigger to plot 2 random lightcurves
        cmd = (
            "python run.py --dump_dir tests/dump_visual --plot_lcs --model_files  "
            + model_file
            + " --plot_file trigger_random.csv"
        )
        call_cmd(cmd)

        lc_files = glob.glob(dir_data_path + "lightcurves/*/early_prediction/*.png")
        assert len(lc_files) == 2
