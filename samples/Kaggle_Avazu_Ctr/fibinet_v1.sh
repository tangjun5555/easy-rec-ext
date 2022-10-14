#!/bin/bash

if [ -z ${easy_rec_ext_data_dir} ]; then
  echo "easy_rec_ext_data_dir is not exists"
  exit 1
else
  echo "easy_rec_ext_data_dir=${easy_rec_ext_data_dir}"
fi

rm -rf fibinet_v1

# Saving dict for global step 25196: auc = 0.71428937, gauc = 0.70688045, global_step = 25196, loss = 0.41361886, loss/loss/cross_entropy_loss = 0.41361886, loss/loss/total_loss = 0.41361886, pcopc = 0.94493103
python -m easy_rec_ext.main --pipeline_config_path=fibinet_v1.json --task_type=train_and_evaluate \
  --train_input_path=${easy_rec_ext_data_dir}/sample_27 \
  --eval_input_path=${easy_rec_ext_data_dir}/sample_30
echo "训练模型sample_27"

# Saving dict for global step 66503: auc = 0.72979456, gauc = 0.7183288, global_step = 66503, loss = 0.40628868, loss/loss/cross_entropy_loss = 0.40628868, loss/loss/total_loss = 0.40628868, pcopc = 0.931607
python -m easy_rec_ext.main --pipeline_config_path=fibinet_v1.json --task_type=train_and_evaluate \
  --train_input_path=${easy_rec_ext_data_dir}/sample_28 \
  --eval_input_path=${easy_rec_ext_data_dir}/sample_30
echo "训练模型sample_28"

# Saving dict for global step 96446: auc = 0.7384362, gauc = 0.7252884, global_step = 96446, loss = 0.40130416, loss/loss/cross_entropy_loss = 0.40130416, loss/loss/total_loss = 0.40130416, pcopc = 0.9943365
python -m easy_rec_ext.main --pipeline_config_path=fibinet_v1.json --task_type=train_and_evaluate \
  --train_input_path=${easy_rec_ext_data_dir}/sample_29 \
  --eval_input_path=${easy_rec_ext_data_dir}/sample_30
echo "训练模型sample_29"

python -m easy_rec_ext.tools.check_mdoel_variable --checkpoint_path=fibinet_v1/ckpt
