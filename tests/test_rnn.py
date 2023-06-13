import os
import pytest
from pathlib import Path
import glob
import shutil
# from tests.test_utils import call_cmd
# from tests.test_utils import testmanager
from .test_utils import call_cmd, testmanager

dir_path = os.path.dirname(os.path.realpath(__file__)) + "/dump/"
dir_data_path = os.path.dirname(os.path.realpath(__file__)) + "/dump_data/"

def test_database():
    cmd = (
        f"python run.py --data --dump_dir tests/dump --raw_dir tests/raw"
    )
    call_cmd(cmd)
    
    # check whether database has been generated 
    assert os.path.exists(dir_path + "processed/database.h5") is True

def test_database_filter():
    cmd = (
        f"python run.py --data --dump_dir tests/dump_data --raw_dir tests/raw "
        f"--list_filters g r"
    )
    call_cmd(cmd)
    
    # check whether database has been generated 
    assert os.path.exists(dir_data_path + "processed/database.h5") is True
    # remove dump_data folder after assertion
    shutil.rmtree(dir_data_path)

def test_database_sntypes():
    cmd = (
        f"python run.py --data --dump_dir tests/dump_data --raw_dir tests/raw "
        f"--sntypes 101"
    )
    call_cmd(cmd)
    
    # check whether database has been generated 
    assert os.path.exists(dir_data_path + "processed/database.h5") is True
    # remove dump_data folder after assertion
    shutil.rmtree(dir_data_path)

def test_rnn_train():

    cmd = (
        f"python run.py --train_rnn --dump_dir tests/dump "
    )
    
    call_cmd(cmd)

    model_dir = dir_path + "models/"
    files = ["*/*.pt", "*/PRED*", "*/METRICS*", "*/train_and_val_loss*", "*/training_log.json"]
    for fi in files:
        assert len([e for e in (Path(model_dir)).glob(fi)]) == 1

def test_rnn_nbclass():

    cmd = (
        f"python run.py --train_rnn --dump_dir tests/dump --nb_classes 2"
        
    )
    
    call_cmd(cmd)

    model_dir = dir_path + "models/"
    files = ["*/*.pt", "*/PRED*", "*/METRICS*", "*/train_and_val_loss*", "*/training_log.json"]
    for fi in files:
        assert len([e for e in (Path(model_dir)).glob(fi)]) == 1


def test_rnn_validate():
    """running rnn_validate generating two files PRED* and METRICS*,
       which are the same filename as train_rnn and cannot be tested 
       by files exist assertion 
       to do: either change the file names or check contents of the file
    """
    cmd = (
        f"python run.py --validate_rnn --dump_dir tests/dump"
    )
    
    call_cmd(cmd)

    # model_dir = dir_path + "models/"
    # files = ["*/PRED*", "*/METRICS*"]
    # for fi in files:
    #     assert len([e for e in (Path(model_dir)).glob(fi)]) == 1

def test_rnn_speed():
    cmd = (
        f"python run.py --validate_rnn --speed --dump_dir tests/dump"
    )
    
    call_cmd(cmd)

def test_rnn_metrics():
    model_dir = dir_path + "models/"
    pred_files = [e for e in (Path(model_dir)).glob("*/PRED*")]
    for pf in pred_files:
        cmd = (
            f"python run.py --validate_rnn --metrics --dump_dir tests/dump "
            f"--prediction_files {pf}"
        )
    
        call_cmd(cmd)

def test_rnn_mfile():
    model_dir = dir_path + "models/"
    model_files = [e for e in (Path(model_dir)).glob("*/*.pt")]
    for mf in model_files:
        cmd = (
            f"python run.py --validate_rnn --dump_dir tests/dump "
            f"--model_files {mf}"
        )

        call_cmd(cmd)

def test_plot_lcs():
    model_dir = dir_path + "models/"
    model_files = [e for e in (Path(model_dir)).glob("*/*.pt")]
    for mf in model_files:
        cmd = (
            f"python run.py --plot_lcs --dump_dir tests/dump/ "
            f"--model_files {mf}"
        )

        call_cmd(cmd)

'''    

   
if __name__ == "__main__":

    test_rnn()
'''