#!/bin/bash

if [ -z ${easy_rec_ext_data_dir} ]; then
  echo "easy_rec_ext_data_dir is not exists"
  exit 1
else
  echo "easy_rec_ext_data_dir=${easy_rec_ext_data_dir}"
fi

python build_candidate_item.py --input_path=${easy_rec_ext_data_dir}/UserBehavior.csv --output_path=${easy_rec_ext_data_dir}/candidate_item
echo "完成构建候选物品特征"

python build_sample.py --input_path=${easy_rec_ext_data_dir}/UserBehavior.csv --output_path=${easy_rec_ext_data_dir}/sample --seq_max_length=50
echo "完成构建训练样本"

tail -n 10000 ${easy_rec_ext_data_dir}/sample_eval.txt > ${easy_rec_ext_data_dir}/sample_eval_1w.txt
python -m easy_rec_ext.main --pipeline_config_path=deep_v1.json --task_type=train_and_evaluate \
  --train_input_path=${easy_rec_ext_data_dir}/sample_train.txt \
  --eval_input_path=${easy_rec_ext_data_dir}/sample_eval_1w.txt
echo "完成模型训练"

python -m easy_rec_ext.main --pipeline_config_path=deep_v1.json --task_type=export
echo "完成模型导出"

version=`ls deep_v1/export | tail -n 1`
python evaluate_model.py --model_pb_path=deep_v1/export/${version} \
  --item_fea_path=${easy_rec_ext_data_dir}/candidate_item_label.txt \
  --eval_sample_path=${easy_rec_ext_data_dir}/sample_eval.txt \
  --vector_dim=100 --topks=5,20,100
python evaluate_model.py --model_pb_path=deep_v1/export/${version} \
  --item_fea_path=${easy_rec_ext_data_dir}/candidate_item_hot.txt \
  --eval_sample_path=${easy_rec_ext_data_dir}/sample_eval.txt \
  --vector_dim=100 --topks=5,20,100
python evaluate_model.py --model_pb_path=deep_v1/export/${version} \
  --item_fea_path=${easy_rec_ext_data_dir}/candidate_item_all.txt \
  --eval_sample_path=${easy_rec_ext_data_dir}/sample_eval.txt \
  --vector_dim=100 --topks=5,20,100
echo "完成模型评估"
