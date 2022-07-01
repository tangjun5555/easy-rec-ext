#!/bin/bash

if [ -z ${easy_rec_ext_data_dir} ]; then
  echo "easy_rec_ext_data_dir is not exists"
  exit 1
else
  echo "easy_rec_ext_data_dir=${easy_rec_ext_data_dir}"
fi

python build_all_item_feature.py --input_path=${easy_rec_ext_data_dir}/UserBehavior.csv --output_path=${easy_rec_ext_data_dir}/all_item_fea.txt
echo "完成构建全库物品特征"

python process_raw_data.py --input_path=${easy_rec_ext_data_dir}/UserBehavior.csv --output_dir=${easy_rec_ext_data_dir} --seq_max_length=50
echo "完成构建训练样本"

head -n 10000 ${easy_rec_ext_data_dir}/sample_20171203.csv > ${easy_rec_ext_data_dir}/sample_20171203_1w.csv
python -m easy_rec_ext.main --pipeline_config_path=deep_v1.json --task_type=train_and_evaluate \
  --train_input_path=${easy_rec_ext_data_dir}/sample_20171201.csv \
  --eval_input_path=${easy_rec_ext_data_dir}/sample_20171203_1w.csv
python -m easy_rec_ext.main --pipeline_config_path=deep_v1.json --task_type=train_and_evaluate \
  --train_input_path=${easy_rec_ext_data_dir}/sample_20171202.csv \
  --eval_input_path=${easy_rec_ext_data_dir}/sample_20171203_1w.csv
echo "完成模型训练"

python -m easy_rec_ext.main --pipeline_config_path=deep_v1.json --task_type=export
echo "完成模型导出"

version=`ls deep_v1/export | tail -n 1`
python evaluate_model.py --model_pb_path=deep_v1/export/${version} \
  --item_fea_path=${easy_rec_ext_data_dir}/all_item_fea.txt \
  --eval_sample_path=${easy_rec_ext_data_dir}/sample_20171203.csv \
  --vector_dim=100 --topks=5,20,100
echo "完成模型评估"
