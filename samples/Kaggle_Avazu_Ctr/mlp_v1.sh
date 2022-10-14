#!/bin/bash

if [ -z ${easy_rec_ext_data_dir} ]; then
  echo "easy_rec_ext_data_dir is not exists"
  exit 1
else
  echo "easy_rec_ext_data_dir=${easy_rec_ext_data_dir}"
fi

rm -rf mlp_v1

# Saving dict for global step 25196: auc = 0.7243722, gauc = 0.71270484, global_step = 25196, loss = 0.4080702, loss/loss/cross_entropy_loss = 0.4080702, loss/loss/total_loss = 0.4080702, pcopc = 1.000718
python -m easy_rec_ext.main --pipeline_config_path=mlp_v1.json --task_type=train_and_evaluate \
  --train_input_path=${easy_rec_ext_data_dir}/sample_27 \
  --eval_input_path=${easy_rec_ext_data_dir}/sample_30
echo "训练模型sample_27"

# Saving dict for global step 66503: auc = 0.73587704, gauc = 0.7224305, global_step = 66503, loss = 0.4047086, loss/loss/cross_entropy_loss = 0.4047086, loss/loss/total_loss = 0.4047086, pcopc = 0.9188919
python -m easy_rec_ext.main --pipeline_config_path=mlp_v1.json --task_type=train_and_evaluate \
  --train_input_path=${easy_rec_ext_data_dir}/sample_28 \
  --eval_input_path=${easy_rec_ext_data_dir}/sample_30
echo "训练模型sample_28"

# Saving dict for global step 96446: auc = 0.74356884, gauc = 0.73021245, global_step = 96446, loss = 0.39867517, loss/loss/cross_entropy_loss = 0.39867517, loss/loss/total_loss = 0.39867517, pcopc = 1.0072732
python -m easy_rec_ext.main --pipeline_config_path=mlp_v1.json --task_type=train_and_evaluate \
  --train_input_path=${easy_rec_ext_data_dir}/sample_29 \
  --eval_input_path=${easy_rec_ext_data_dir}/sample_30
echo "训练模型sample_29"
