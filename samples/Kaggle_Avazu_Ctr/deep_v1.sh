#!/bin/bash

if [ -z ${easy_rec_ext_data_dir} ]; then
  echo "easy_rec_ext_data_dir is not exists"
  exit 1
else
  echo "easy_rec_ext_data_dir=${easy_rec_ext_data_dir}"
fi

rm -rf deep_v1

python -m easy_rec_ext.main --pipeline_config_path=deep_v1.json --task_type=train_and_evaluate \
  --train_input_path=${easy_rec_ext_data_dir}/sample_27 \
  --eval_input_path=${easy_rec_ext_data_dir}/sample_30
echo "训练模型sample_27"

python -m easy_rec_ext.main --pipeline_config_path=deep_v1.json --task_type=train_and_evaluate \
  --train_input_path=${easy_rec_ext_data_dir}/sample_28 \
  --eval_input_path=${easy_rec_ext_data_dir}/sample_30
echo "训练模型sample_28"

python -m easy_rec_ext.main --pipeline_config_path=deep_v1.json --task_type=train_and_evaluate \
  --train_input_path=${easy_rec_ext_data_dir}/sample_29 \
  --eval_input_path=${easy_rec_ext_data_dir}/sample_30
echo "训练模型sample_29"
