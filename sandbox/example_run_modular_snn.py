import os
import supernnova.conf as conf
from supernnova.data import make_dataset
from supernnova.training import train_rnn
from supernnova.validation import validate_rnn

"""Example for running SuperNNova as a module

if installed by "pip install supernnova"
you can run this code in the parent folder (where run.py is)
"""

# get config args
args =  conf.get_args()

# create database
args.data = True 						# conf: making new dataset
args.dump_dir = "tests/dump" 			# conf: where the dataset will be saved
args.raw_dir = "tests/raw"				# conf: where raw photometry files are saved 
args.fits_dir = "tests/fits"			# conf: where salt2fits are saved 
settings = conf.get_settings(args) 		# conf: set settings
make_dataset.make_dataset(settings)		# make dataset

# train model
args.data = False						# conf: no database creation
args.train_rnn = True					# conf: train rnn
args.dump_dir = "tests/dump" 			# conf: where the dataset is saved
args.nb_epoch = 2						# conf: training epochs
settings = conf.get_settings(args) 		# conf: set settings
train_rnn.train(settings)				# train rnn

# validate (test set classificatio)
args.data = False						# conf: no database creation
args.train_rnn = False					# conf: no train rnn
args.validate_rnn = False				# conf: validate rnn
args.dump_dir = "tests/dump" 			# conf: where the dataset is saved
settings = conf.get_settings(args) 		# conf: set settings
validate_rnn.get_predictions(settings)	# classify test set