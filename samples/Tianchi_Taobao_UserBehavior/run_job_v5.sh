#!/bin/bash

# set local data path
easy_rec_ext_data_dir=/Users/jun.tang6/Data/Tianchi_UserBehavior
if [ -z ${easy_rec_ext_data_dir} ]; then
  echo "easy_rec_ext_data_dir is not exists"
  exit 1
else
  echo "easy_rec_ext_data_dir=${easy_rec_ext_data_dir}"
fi

python build_candidate_item.py --input_path=${easy_rec_ext_data_dir}/UserBehavior.csv --output_path=${easy_rec_ext_data_dir}/candidate_item
echo "完成构建候选物品特征"

python build_sample_v5.py --input_path=${easy_rec_ext_data_dir}/UserBehavior.csv --output_path=${easy_rec_ext_data_dir}/sample_v5 --all_item_path=${easy_rec_ext_data_dir}/candidate_item_all.txt --user_train_neg_num=1 --st_seq_max_length=10 --lt_seq_max_length=30 --conversion_seq_max_length=10
tail -n 100000 ${easy_rec_ext_data_dir}/sample_v5_train_20171202.txt > ${easy_rec_ext_data_dir}/sample_v5_train_20171202_10w.txt
head -n 50000 ${easy_rec_ext_data_dir}/sample_v5_eval.txt > ${easy_rec_ext_data_dir}/sample_v5_eval_5w.txt
echo "完成构建训练样本"

rm -rf deep_v5_dssm
rm -rf deep_v5_sdm
for slice in 20171130 20171201 20171202
do
echo "当前样本:${slice}"
python -m easy_rec_ext.main --pipeline_config_path=deep_v5_dssm.json --task_type=train_and_evaluate \
  --train_input_path=${easy_rec_ext_data_dir}/sample_v5_train_${slice}.txt \
  --eval_input_path=${easy_rec_ext_data_dir}/sample_v5_train_20171202_10w.txt
python -m easy_rec_ext.main --pipeline_config_path=deep_v5_sdm.json --task_type=train_and_evaluate \
  --train_input_path=${easy_rec_ext_data_dir}/sample_v5_train_${slice}.txt \
  --eval_input_path=${easy_rec_ext_data_dir}/sample_v5_train_20171202_10w.txt
done
echo "完成模型训练"

python -m easy_rec_ext.main --pipeline_config_path=deep_v5_dssm.json --task_type=export
python -m easy_rec_ext.main --pipeline_config_path=deep_v5_sdm.json --task_type=export
echo "完成模型导出"

dssm_version=`ls deep_v5_dssm/export | tail -n 1`
#评估样本数量: 1374393
#评估指标recall@100:0.014798532879605761
#评估指标recall@200:0.02248701790535895
python evaluate_model.py \
  --model_pb_path=deep_v5_dssm/export/${dssm_version} \
  --item_fea_path=${easy_rec_ext_data_dir}/candidate_item_all.txt \
  --eval_sample_path=${easy_rec_ext_data_dir}/sample_v5_eval.txt \
  --vector_dim=50 \
  --topks=100,200 \
  --default_user_fea="1000095,365332|3163393|365332|1788072,3002561|982926|3002561|4217906,1|2|3|4,3272707|3619589|4243746|2654231|897689|2150511|2929806|1707899|5047770|1738475|4841993|3937094|2936164|4835167|1839704|1228807|2002647|810130|1632861|749811|4869914|3612389|2352497|2084614|3787412|3143296|3686404|1150409|4420114|1063918,2355072|998114|998114|3744190|3744190|4615159|3744190|4615159|5018044|4615159|4615159|3744190|3744190|4615159|3744190|3744190|1521931|1922644|4801426|4145813|4145813|3524510|1973012|4163659|874415|810632|1787510|5045733|3524510|3524510,1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30,4581289|4137245|4843948,4615159|4145813|1464116,1|2|3" \
  --default_item_fea="2041056,4801426"
echo "完成dssm模型评估"

sdm_version=`ls deep_v5_sdm/export | tail -n 1`
#评估样本数量: 1374393
#评估指标recall@100:0.015334042009818152
#评估指标recall@200:0.023884725838970368
python evaluate_model.py \
  --model_pb_path=deep_v5_sdm/export/${sdm_version} \
  --item_fea_path=${easy_rec_ext_data_dir}/candidate_item_all.txt \
  --eval_sample_path=${easy_rec_ext_data_dir}/sample_v5_eval.txt \
  --vector_dim=50 \
  --topks=100,200 \
  --default_user_fea="1000095,365332|3163393|365332|1788072,3002561|982926|3002561|4217906,1|2|3|4,3272707|3619589|4243746|2654231|897689|2150511|2929806|1707899|5047770|1738475|4841993|3937094|2936164|4835167|1839704|1228807|2002647|810130|1632861|749811|4869914|3612389|2352497|2084614|3787412|3143296|3686404|1150409|4420114|1063918,2355072|998114|998114|3744190|3744190|4615159|3744190|4615159|5018044|4615159|4615159|3744190|3744190|4615159|3744190|3744190|1521931|1922644|4801426|4145813|4145813|3524510|1973012|4163659|874415|810632|1787510|5045733|3524510|3524510,1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30,4581289|4137245|4843948,4615159|4145813|1464116,1|2|3" \
  --default_item_fea="2041056,4801426"
echo "完成sdm模型评估"
