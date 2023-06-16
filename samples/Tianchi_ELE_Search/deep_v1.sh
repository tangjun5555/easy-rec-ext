#!/bin/bash

if [ -z ${easy_rec_ext_data_dir} ]; then
  echo "easy_rec_ext_data_dir is not exists"
  exit 1
else
  echo "easy_rec_ext_data_dir=${easy_rec_ext_data_dir}"
fi

# Saving dict for global step 3243: auc = 0.744025, gauc = 0.73820055, global_step = 3243, loss = 0.2805507, loss/loss/cross_entropy_loss = 0.2805507, loss/loss/regularization_loss = 0.0, loss/loss/total_loss = 0.2805507, pcopc = 0.98046625
python -m easy_rec_ext.main --pipeline_config_path=deep_v1.json --task_type=train_and_evaluate --train_input_path=${easy_rec_ext_data_dir}/tianchi_public_data_train_new.txt --eval_input_path=${easy_rec_ext_data_dir}/tianchi_public_data_test_new.txt
echo "完成模型训练"

python -m easy_rec_ext.main --pipeline_config_path=deep_v1.json --task_type=export
echo "完成模型导出"
