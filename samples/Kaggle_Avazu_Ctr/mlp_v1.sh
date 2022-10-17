#!/bin/bash

if [ -z ${easy_rec_ext_data_dir} ]; then
  echo "easy_rec_ext_data_dir is not exists"
  exit 1
else
  echo "easy_rec_ext_data_dir=${easy_rec_ext_data_dir}"
fi

rm -rf mlp_v1

for part in 21 22 23 24 25 26 27 28 29
do
python -m easy_rec_ext.main --pipeline_config_path=mlp_v1.json --task_type=train_and_evaluate \
  --train_input_path=${easy_rec_ext_data_dir}/sample_${part} \
  --eval_input_path=${easy_rec_ext_data_dir}/sample_30
echo "训练模型sample_${part}"
done
# Saving dict for global step 32211: auc = 0.7224633, gauc = 0.71208155, global_step = 32211, loss = 0.40958744, loss/loss/cross_entropy_loss = 0.40958744, loss/loss/total_loss = 0.40958744, pcopc = 1.0090172
# Saving dict for global step 73908: auc = 0.7201729, gauc = 0.70876056, global_step = 73908, loss = 0.41331875, loss/loss/cross_entropy_loss = 0.41331875, loss/loss/total_loss = 0.41331875, pcopc = 0.8714226
# Saving dict for global step 104149: auc = 0.73032904, gauc = 0.7183032, global_step = 104149, loss = 0.4060962, loss/loss/cross_entropy_loss = 0.4060962, loss/loss/total_loss = 0.4060962, pcopc = 0.95460725
# Saving dict for global step 130207: auc = 0.7254425, gauc = 0.71483237, global_step = 130207, loss = 0.4171358, loss/loss/cross_entropy_loss = 0.4171358, loss/loss/total_loss = 0.4171358, pcopc = 0.7330842
# Saving dict for global step 156482: auc = 0.73521745, gauc = 0.7230974, global_step = 156482, loss = 0.40355298, loss/loss/cross_entropy_loss = 0.40355298, loss/loss/total_loss = 0.40355298, pcopc = 0.95498216
# Saving dict for global step 186450: auc = 0.73524964, gauc = 0.7228316, global_step = 186450, loss = 0.4038192, loss/loss/cross_entropy_loss = 0.4038192, loss/loss/total_loss = 0.4038192, pcopc = 0.9815651
# Saving dict for global step 211646: auc = 0.73850733, gauc = 0.7253919, global_step = 211646, loss = 0.4012659, loss/loss/cross_entropy_loss = 0.4012659, loss/loss/total_loss = 0.4012659, pcopc = 1.0020953
# Saving dict for global step 252953: auc = 0.74210286, gauc = 0.7282094, global_step = 252953, loss = 0.40046012, loss/loss/cross_entropy_loss = 0.40046012, loss/loss/total_loss = 0.40046012, pcopc = 0.92782956
# Saving dict for global step 282896: auc = 0.74720675, gauc = 0.733077, global_step = 282896, loss = 0.39685553, loss/loss/cross_entropy_loss = 0.39685553, loss/loss/total_loss = 0.39685553, pcopc = 0.9438787

python -m easy_rec_ext.tools.check_model_variable --checkpoint_path=mlp_v1/ckpt
