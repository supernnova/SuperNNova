#!/bin/bash

source activate pytorch

cd /home/ubuntu/SuperNNova
echo "USING small time window"
python run.py --dump_dir /home/ubuntu/dump_elasticc/  --list_filters u g r i z Y --sntypes '{"10": "Ia", "11": "Ia", "12" : "Ia", "20": "Ia", "21": "Ia", "25": "Ia", "26": "Ia", "27": "Ia", "30": "Ia", "31": "Ia", "32": "Ia", "35": "Ia", "36": "Ia", "37": "Ia", "40": "Long", "42": "Long", "45": "Long", "46": "Long", "50":"Fast", "51": "Fast", "59" : "Long", "60" : "Recurring", "80" : "Recurring", "82" : "Recurring", "83" : "Recurring", "84" : "Recurring", "87": "Fast", "88": "Fast", "89" : "Fast", "90" : "Recurring", "91" : "Recurring"}' --sntype_var SIM_TYPE_INDEX  --train_rnn --redshift zspe --batch_size 240 --plot_lcs --use_cuda
echo "train over"