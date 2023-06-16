#!/bin/bash

rm -rf easyrec_din_v1
for (( epoch=1;epoch<=3;epoch++ ))
do
  rm -rf easyrec_din_v1/ESTIMATOR_TRAIN_DONE
  python -m easy_rec.python.train_eval --pipeline_config_path=easyrec_din_v1.config --continue_train --train_input_path=${data_dir}/elm_train.csv --eval_input_path=${data_dir}/elm_eval.csv
  echo "easyrec_din_v1完成${epoch}轮训练"
  # Saving dict for global step 7813: auc = 0.55579066, gauc = 0.53785807, global_step = 7813, loss = 0.526877, loss/loss/cross_entropy_loss = 0.526877, loss/loss/total_loss = 0.526877
  # Saving dict for global step 15626: auc = 0.532865, gauc = 0.5213874, global_step = 15626, loss = 0.84674084, loss/loss/cross_entropy_loss = 0.84674084, loss/loss/total_loss = 0.84674084
  # Saving dict for global step 23439: auc = 0.5277359, gauc = 0.5174528, global_step = 23439, loss = 1.0121183, loss/loss/cross_entropy_loss = 1.0121183, loss/loss/total_loss = 1.0121183
done

rm -rf easyrec_bst_v1
for (( epoch=1;epoch<=3;epoch++ ))
do
  rm -rf easyrec_bst_v1/ESTIMATOR_TRAIN_DONE
  python -m easy_rec.python.train_eval --pipeline_config_path=easyrec_bst_v1.config --continue_train --train_input_path=${data_dir}/elm_train.csv --eval_input_path=${data_dir}/elm_eval.csv
  echo "easyrec_bst_v1完成${epoch}轮训练"

  # Saving dict for global step 7813: auc = 0.56072456, gauc = 0.54233456, global_step = 7813, loss = 0.5191261, loss/loss/cross_entropy_loss = 0.5191261, loss/loss/total_loss = 0.5191261
  # Saving dict for global step 15626: auc = 0.5348352, gauc = 0.5238664, global_step = 15626, loss = 0.7927517, loss/loss/cross_entropy_loss = 0.7927517, loss/loss/total_loss = 0.7927517
  # Saving dict for global step 23439: auc = 0.5296059, gauc = 0.5251349, global_step = 23439, loss = 0.96280956, loss/loss/cross_entropy_loss = 0.96280956, loss/loss/total_loss = 0.96280956
done
