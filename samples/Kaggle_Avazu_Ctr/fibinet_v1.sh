#!/bin/bash

if [ -z ${easy_rec_ext_data_dir} ]; then
  echo "easy_rec_ext_data_dir is not exists"
  exit 1
else
  echo "easy_rec_ext_data_dir=${easy_rec_ext_data_dir}"
fi

rm -rf fibinet_v1

for part in 21 22 23 24 25 26 27 28 29
do
python -m easy_rec_ext.main --pipeline_config_path=fibinet_v1.json --task_type=train_and_evaluate \
  --train_input_path=${easy_rec_ext_data_dir}/sample_${part} \
  --eval_input_path=${easy_rec_ext_data_dir}/sample_30
echo "训练模型sample_${part}"
done
# Saving dict for global step 32211: auc = 0.71557295, gauc = 0.7074479, global_step = 32211, loss = 0.412461, loss/loss/cross_entropy_loss = 0.412461, loss/loss/total_loss = 0.412461, pcopc = 0.9992151
# Saving dict for global step 73908: auc = 0.71057045, gauc = 0.70143414, global_step = 73908, loss = 0.41686013, loss/loss/cross_entropy_loss = 0.41686013, loss/loss/total_loss = 0.41686013, pcopc = 0.9184518
# Saving dict for global step 104149: auc = 0.71678007, gauc = 0.70681196, global_step = 104149, loss = 0.41246614, loss/loss/cross_entropy_loss = 0.41246614, loss/loss/total_loss = 0.41246614, pcopc = 1.0057559
# Saving dict for global step 130207: auc = 0.72092324, gauc = 0.7110829, global_step = 130207, loss = 0.42072004, loss/loss/cross_entropy_loss = 0.42072004, loss/loss/total_loss = 0.42072004, pcopc = 0.7245713
# Saving dict for global step 156482: auc = 0.7272918, gauc = 0.71739185, global_step = 156482, loss = 0.40890497, loss/loss/cross_entropy_loss = 0.40890497, loss/loss/total_loss = 0.40890497, pcopc = 0.93824744
# Saving dict for global step 186450: auc = 0.7270293, gauc = 0.71805066, global_step = 186450, loss = 0.4083485, loss/loss/cross_entropy_loss = 0.4083485, loss/loss/total_loss = 0.4083485, pcopc = 0.96550184
# Saving dict for global step 211646: auc = 0.7319008, gauc = 0.7192324, global_step = 211646, loss = 0.40516135, loss/loss/cross_entropy_loss = 0.40516135, loss/loss/total_loss = 0.40516135, pcopc = 0.99258745
# Saving dict for global step 252953: auc = 0.741235, gauc = 0.72771615, global_step = 252953, loss = 0.40117133, loss/loss/cross_entropy_loss = 0.40117133, loss/loss/total_loss = 0.40117133, pcopc = 0.9232592
# Saving dict for global step 282896: auc = 0.74598646, gauc = 0.7322617, global_step = 282896, loss = 0.398138, loss/loss/cross_entropy_loss = 0.398138, loss/loss/total_loss = 0.398138, pcopc = 0.9455892

python -m easy_rec_ext.tools.check_model_variable --checkpoint_path=fibinet_v1/ckpt
