#!/bin/bash

source activate pytorch

cd /home/ubuntu/SuperNNova
python run.py --dump_dir /home/ubuntu/dump_elasticc/ --raw_dir /home/ubuntu/TRAINING_SAMPLES --list_filters u g r i z Y --phot_reject PHOTFLAG --phot_reject_list 7168 --photo_window_var PEAKMJD --photo_window_min -20 --photo_window_max 100 --sntypes '{"10": "Ia", "11": "Ia", "12" : "Ia", "20": "SNIbc", "21": "SNIbc", "25": "SNIbc", "26": "SNIbc", "27": "SNIbc", "30": "SNII", "31": "SNII", "32": "SNII", "35": "SNII", "36": "SNII", "37": "SNII", "40": "SLSN no_host", "42": "TDE", "45": "ILOT", "46": "CART", "50":"KN", "51": "KN", "59" : "PISN", "60" : "AGN", "80" : "RR Lyrae", "82" : "M dwarf flare", "83" : "EB", "84" : "dwarf novae", "87": "ulens", "88": "ulens", "89" : "ulens", "90" : "Cepheid", "91" : "delta scuti"}' --sntype_var SIM_TYPE_INDEX --data --photo_window_files HEAD 

python run.py --dump_dir /home/ubuntu/dump_elasticc/  --list_filters u g r i z Y --sntypes '{"10": "Ia", "11": "Ia", "12" : "Ia", "20": "SNIbc", "21": "SNIbc", "25": "SNIbc", "26": "SNIbc", "27": "SNIbc", "30": "SNII", "31": "SNII", "32": "SNII", "35": "SNII", "36": "SNII", "37": "SNII", "40": "SLSN no_host", "42": "TDE", "45": "ILOT", "46": "CART", "50":"KN", "51": "KN", "59" : "PISN", "60" : "AGN", "80" : "RR Lyrae", "82" : "M dwarf flare", "83" : "EB", "84" : "dwarf novae", "87": "ulens", "88": "ulens", "89" : "ulens", "90" : "Cepheid", "91" : "delta scuti"}' --sntype_var SIM_TYPE_INDEX  --train_rnn --redshift zspe --batch_size 240 --plot_lcs --use_cuda
