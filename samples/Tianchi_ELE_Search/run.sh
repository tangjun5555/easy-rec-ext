#!/bin/bash

# Saving dict for global step 1622: auc = 0.7439498, gauc = 0.73842174, global_step = 1622, loss = 0.28166285, loss/loss/cross_entropy_loss = 0.28166285, loss/loss/total_loss = 0.28166285
python -m easy_rec.python.train_eval --pipeline_config_path=easyrec_din_v1.config --continue_train --train_input_path=${data_dir}/tianchi_public_data_train_new.txt --eval_input_path=${data_dir}/tianchi_public_data_test_new.txt

# Saving dict for global step 1622: auc = 0.74537975, gauc = 0.73318183, global_step = 1622, loss = 0.28246605, loss/loss/cross_entropy_loss = 0.28246605, loss/loss/total_loss = 0.28246605
python -m easy_rec.python.train_eval --pipeline_config_path=easyrec_bst_v1.config --continue_train --train_input_path=${data_dir}/tianchi_public_data_train_new.txt --eval_input_path=${data_dir}/tianchi_public_data_test_new.txt

