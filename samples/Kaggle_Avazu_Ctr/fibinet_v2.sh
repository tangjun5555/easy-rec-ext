#!/bin/bash

if [ -z ${easy_rec_ext_data_dir} ]; then
  echo "easy_rec_ext_data_dir is not exists"
  exit 1
else
  echo "easy_rec_ext_data_dir=${easy_rec_ext_data_dir}"
fi

rm -rf fibinet_v2

# Saving dict for global step 25196: auc = 0.71251565, gauc = 0.7034642, global_step = 25196, loss = 0.41468763, loss/loss/cross_entropy_loss = 0.41468763, loss/loss/total_loss = 0.41468763, pcopc = 0.919328
python -m easy_rec_ext.main --pipeline_config_path=fibinet_v2.json --task_type=train_and_evaluate \
  --train_input_path=${easy_rec_ext_data_dir}/sample_27 \
  --eval_input_path=${easy_rec_ext_data_dir}/sample_30
echo "训练模型sample_27"

# Saving dict for global step 66503: auc = 0.73008704, gauc = 0.71764696, global_step = 66503, loss = 0.40640232, loss/loss/cross_entropy_loss = 0.40640232, loss/loss/total_loss = 0.40640232, pcopc = 0.9135876
python -m easy_rec_ext.main --pipeline_config_path=fibinet_v2.json --task_type=train_and_evaluate \
  --train_input_path=${easy_rec_ext_data_dir}/sample_28 \
  --eval_input_path=${easy_rec_ext_data_dir}/sample_30
echo "训练模型sample_28"

# Saving dict for global step 96446: auc = 0.73776424, gauc = 0.7237419, global_step = 96446, loss = 0.4018585, loss/loss/cross_entropy_loss = 0.4018585, loss/loss/total_loss = 0.4018585, pcopc = 0.94768304
python -m easy_rec_ext.main --pipeline_config_path=fibinet_v2.json --task_type=train_and_evaluate \
  --train_input_path=${easy_rec_ext_data_dir}/sample_29 \
  --eval_input_path=${easy_rec_ext_data_dir}/sample_30
echo "训练模型sample_29"

python -m easy_rec_ext.tools.check_mdoel_variable --checkpoint_path=fibinet_v2/ckpt
