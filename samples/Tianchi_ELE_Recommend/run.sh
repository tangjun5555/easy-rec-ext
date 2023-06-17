#!/bin/bash

# Saving dict for global step 7813: auc = 0.55579066, gauc = 0.53785807, global_step = 7813, loss = 0.526877, loss/loss/cross_entropy_loss = 0.526877, loss/loss/total_loss = 0.526877
python -m easy_rec.python.train_eval --pipeline_config_path=easyrec_din_v1.config --continue_train --train_input_path=${data_dir}/elm_train.csv --eval_input_path=${data_dir}/elm_eval.csv

# Saving dict for global step 7813: auc = 0.56072456, gauc = 0.54233456, global_step = 7813, loss = 0.5191261, loss/loss/cross_entropy_loss = 0.5191261, loss/loss/total_loss = 0.5191261
python -m easy_rec.python.train_eval --pipeline_config_path=easyrec_bst_v1.config --continue_train --train_input_path=${data_dir}/elm_train.csv --eval_input_path=${data_dir}/elm_eval.csv
