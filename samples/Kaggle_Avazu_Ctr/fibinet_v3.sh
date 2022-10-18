#!/bin/bash

if [ -z ${easy_rec_ext_data_dir} ]; then
  echo "easy_rec_ext_data_dir is not exists"
  exit 1
else
  echo "easy_rec_ext_data_dir=${easy_rec_ext_data_dir}"
fi

rm -rf fibinet_v3

for part in 21 22 23 24 25 26 27 28 29
do
python -m easy_rec_ext.main --pipeline_config_path=fibinet_v3.json --task_type=train_and_evaluate \
  --train_input_path=${easy_rec_ext_data_dir}/sample_${part} \
  --eval_input_path=${easy_rec_ext_data_dir}/sample_30
echo "训练模型sample_${part}"
done

python -m easy_rec_ext.tools.check_model_variable --checkpoint_path=fibinet_v3/ckpt
