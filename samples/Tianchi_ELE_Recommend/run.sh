#!/bin/bash

rm -rf easyrec_din_v1
for (( epoch=1;epoch<=3;epoch++ ))
do
  rm -rf easyrec_din_v1/ESTIMATOR_TRAIN_DONE
  python -m easy_rec.python.train_eval --pipeline_config_path=easyrec_din_v1.config --continue_train --train_input_path=${data_dir}/elm_train.csv --eval_input_path=${data_dir}/elm_eval.csv
  echo "easyrec_din_v1完成${epoch}轮训练"
done

rm -rf easyrec_bst_v1
for (( epoch=1;epoch<=3;epoch++ ))
do
  rm -rf easyrec_bst_v1/ESTIMATOR_TRAIN_DONE
  python -m easy_rec.python.train_eval --pipeline_config_path=easyrec_bst_v1.config --continue_train --train_input_path=${data_dir}/elm_train.csv --eval_input_path=${data_dir}/elm_eval.csv
  echo "easyrec_bst_v1完成${epoch}轮训练"
done
