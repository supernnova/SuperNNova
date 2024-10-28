import supernnova.conf as conf
from supernnova.data import make_dataset
from supernnova.training import train_rnn
from supernnova.validation import validate_rnn
import os

"""Example for running SuperNNova as a module

if installed by "pip install supernnova"
you can run this code in the parent folder (where tests/ is)
"""
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(current_dir)

if __name__ == "__main__":
    # get config args
    command_arg = "make_data"
    args = conf.get_args(command_arg)

    # create database
    args.dump_dir = repo_dir + "/tests/dump"  # conf: where the dataset will be saved
    args.raw_dir = repo_dir + "/tests/raw"  # conf: where raw photometry files are saved
    args.fits_dir = repo_dir + "/tests/fits"  # conf: where salt2fits are saved
    settings = conf.get_settings(command_arg, args)  # conf: set settings
    make_dataset.make_dataset(settings)  # make dataset

    # get config args
    command_arg = "train_rnn"
    args = conf.get_args(command_arg)

    # train rnn
    args.dump_dir = repo_dir + "/tests/dump"  # conf: where the dataset is saved
    args.nb_epoch = 2  # conf: training epochs
    settings = conf.get_settings(command_arg, args)  # conf: set settings
    train_rnn.train(settings)  # train rnn

    # get config args
    command_arg = "validate_rnn"
    args = conf.get_args(command_arg)

    # validate rnn
    args.dump_dir = repo_dir + "/tests/dump"  # conf: where the dataset is saved
    settings = conf.get_settings(command_arg, args)  # conf: set settings
    validate_rnn.get_predictions(settings)  # classify test set
