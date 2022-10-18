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
# Saving dict for global step 32211: auc = 0.7258651, gauc = 0.71619207, global_step = 32211, loss = 0.40864724, loss/loss/cross_entropy_loss = 0.40864724, loss/loss/total_loss = 0.40864724, pcopc = 1.0061632
# Saving dict for global step 73908: auc = 0.7232687, gauc = 0.7114876, global_step = 73908, loss = 0.41099375, loss/loss/cross_entropy_loss = 0.41099375, loss/loss/total_loss = 0.41099375, pcopc = 0.8842273
# Saving dict for global step 104149: auc = 0.7300392, gauc = 0.7177111, global_step = 104149, loss = 0.40602404, loss/loss/cross_entropy_loss = 0.40602404, loss/loss/total_loss = 0.40602404, pcopc = 0.98693657
# Saving dict for global step 130207: auc = 0.7309687, gauc = 0.7200182, global_step = 130207, loss = 0.41395912, loss/loss/cross_entropy_loss = 0.41395912, loss/loss/total_loss = 0.41395912, pcopc = 0.7400584
# Saving dict for global step 156482: auc = 0.73634446, gauc = 0.7242361, global_step = 156482, loss = 0.40282685, loss/loss/cross_entropy_loss = 0.40282685, loss/loss/total_loss = 0.40282685, pcopc = 0.94964725
# Saving dict for global step 186450: auc = 0.73754114, gauc = 0.7252248, global_step = 186450, loss = 0.40241668, loss/loss/cross_entropy_loss = 0.40241668, loss/loss/total_loss = 0.40241668, pcopc = 0.95454425
# Saving dict for global step 211646: auc = 0.7389662, gauc = 0.72583187, global_step = 211646, loss = 0.4012984, loss/loss/cross_entropy_loss = 0.4012984, loss/loss/total_loss = 0.4012984, pcopc = 0.94675076
# Saving dict for global step 252953: auc = 0.7428109, gauc = 0.72922444, global_step = 252953, loss = 0.39942393, loss/loss/cross_entropy_loss = 0.39942393, loss/loss/total_loss = 0.39942393, pcopc = 0.94714624
# Saving dict for global step 282896: auc = 0.7452766, gauc = 0.73152685, global_step = 282896, loss = 0.39831087, loss/loss/cross_entropy_loss = 0.39831087, loss/loss/total_loss = 0.39831087, pcopc = 0.9283987

python -m easy_rec_ext.tools.check_model_variable --checkpoint_path=mlp_v1/ckpt
