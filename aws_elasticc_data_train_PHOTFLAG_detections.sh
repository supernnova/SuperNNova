#!/bin/bash

source activate pytorch

cd /home/ubuntu/SuperNNova
echo "data start"
python run.py --dump_dir ./dump_elasticc_photflag_detections/ --raw_dir /home/ubuntu/TRAINING_SAMPLES --list_filters u g r i z Y --phot_reject PHOTFLAG --phot_reject_list 7168 0 --sntypes '{"10": "Ia", "11": "Ia", "12" : "Ia", "20": "SNIbc", "21": "SNIbc", "25": "SNIbc", "26": "SNIbc", "27": "SNIbc", "30": "SNII", "31": "SNII", "32": "SNII", "35": "SNII", "36": "SNII", "37": "SNII", "40": "SLSN no_host", "42": "TDE", "45": "ILOT", "46": "CART", "50":"KN", "51": "KN", "59" : "PISN", "60" : "AGN", "80" : "RR Lyrae", "82" : "M dwarf flare", "83" : "EB", "84" : "dwarf novae", "87": "ulens", "88": "ulens", "89" : "ulens", "90" : "Cepheid", "91" : "delta scuti"}' --sntype_var SIM_TYPE_INDEX --data 
echo "data over"

echo "with redshift"
python run.py --dump_dir ./dump_elasticc_photflag_detections/ --list_filters u g r i z Y --sntypes '{"10": "Ia", "11": "Ia", "12" : "Ia", "20": "SNIbc", "21": "SNIbc", "25": "SNIbc", "26": "SNIbc", "27": "SNIbc", "30": "SNII", "31": "SNII", "32": "SNII", "35": "SNII", "36": "SNII", "37": "SNII", "40": "SLSN no_host", "42": "TDE", "45": "ILOT", "46": "CART", "50":"KN", "51": "KN", "59" : "PISN", "60" : "AGN", "80" : "RR Lyrae", "82" : "M dwarf flare", "83" : "EB", "84" : "dwarf novae", "87": "ulens", "88": "ulens", "89" : "ulens", "90" : "Cepheid", "91" : "delta scuti"}' --sntype_var SIM_TYPE_INDEX --train_rnn --use_cuda --redshift zspe --plot_lcs

echo "train over"
