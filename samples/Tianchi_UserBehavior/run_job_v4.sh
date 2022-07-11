#!/bin/bash

if [ -z ${easy_rec_ext_data_dir} ]; then
  echo "easy_rec_ext_data_dir is not exists"
  exit 1
else
  echo "easy_rec_ext_data_dir=${easy_rec_ext_data_dir}"
fi

python build_candidate_item.py --input_path=${easy_rec_ext_data_dir}/UserBehavior.csv --output_path=${easy_rec_ext_data_dir}/candidate_item
echo "完成构建候选物品特征"

python build_sample_v4.py --input_path=${easy_rec_ext_data_dir}/UserBehavior.csv --output_path=${easy_rec_ext_data_dir}/sample_v4 --all_item_path=${easy_rec_ext_data_dir}/candidate_item_all.txt --user_train_pos_num=20 --user_train_neg_num=1 --st_seq_max_length=20 --lt_seq_max_length=50 --conversion_seq_max_length=10
tail -n 50000 ${easy_rec_ext_data_dir}/sample_v4_eval.txt > ${easy_rec_ext_data_dir}/sample_v4_eval_5w.txt
tail -n 100000 ${easy_rec_ext_data_dir}/sample_v4_train_19.txt > ${easy_rec_ext_data_dir}/sample_v4_train_19_10w.txt
echo "完成构建训练样本"

rm -rf deep_v4_dssm
rm -rf deep_v4_sdm
for slice in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19
do
echo "当前样本:${slice}"
python -m easy_rec_ext.main --pipeline_config_path=deep_v4_dssm.json --task_type=train_and_evaluate \
  --train_input_path=${easy_rec_ext_data_dir}/sample_v4_train_${slice}.txt \
  --eval_input_path=${easy_rec_ext_data_dir}/sample_v4_train_19_10w.txt
python -m easy_rec_ext.main --pipeline_config_path=deep_v4_sdm.json --task_type=train_and_evaluate \
  --train_input_path=${easy_rec_ext_data_dir}/sample_v4_train_${slice}.txt \
  --eval_input_path=${easy_rec_ext_data_dir}/sample_v4_train_19_10w.txt
done
echo "完成模型训练"

python -m easy_rec_ext.main --pipeline_config_path=deep_v4_dssm.json --task_type=export
python -m easy_rec_ext.main --pipeline_config_path=deep_v4_sdm.json --task_type=export
echo "完成模型导出"

dssm_version=`ls deep_v4_dssm/export | tail -n 1`
#评估样本数量: 50000
#评估指标recall@10:0.00104
#评估指标recall@100:0.00642
#评估指标recall@200:0.01104
python evaluate_model.py --model_pb_path=deep_v4_dssm/export/${dssm_version} \
  --item_fea_path=${easy_rec_ext_data_dir}/candidate_item_all.txt \
  --eval_sample_path=${easy_rec_ext_data_dir}/sample_v4_eval_5w.txt \
  --vector_dim=50 --topks=10,100,200 \
  --default_user_fea="1,271696|818610|4954999|929177|2278603|3219016|2028434,411153|411153|411153|4801426|3002561|3002561|4801426,2|2|3|4|4|4|4,1|2|3|4|5|6|7,2104483|3219016|2041056|4954999|4198227|1323189|4666650|79715|4954999|3682069|4170517|3911125|1340922|4170517|2087357|3157558|2087357|4973305|46259|3239041|2791761|1305059|4092065|266784|46259|266784|4152983|4615417|3239041|5002615|2734026|5002615|2286574|1338525|3108797|2951368|2266567|1531036|3745169|3827899|230380|4606018|4365585|3830808|2576651|2333346|2268318,4756105|3002561|4801426|411153|1320293|3524510|4756105|2355072|411153|4690421|149192|982926|4690421|149192|2131531|2520771|2131531|2520771|149192|2355072|2355072|2520771|2355072|2520771|149192|2520771|2355072|4145813|2355072|2520377|4145813|2520377|2465336|149192|2355072|1080785|4145813|2920476|2891509|2920476|411153|2735466|2520377|4181361|149192|2520771|2520377,5|5|5|5|5|5|6|6|6|6|6|6|6|7|7|7|7|7|7|7|8|8|8|8|8|8|8|8|8|9|9|9|9|9|10|10|10|10|10|10|11|11|11|11|12|12|12,1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47,-1,-1,-1,-1" \
  --default_item_fea="2041056,4801426"
echo "完成dssm模型评估"

sdm_version=`ls deep_v4_sdm/export | tail -n 1`
#评估样本数量: 50000
#评估指标recall@10:0.00248
#评估指标recall@100:0.01234
#评估指标recall@200:0.0183
python evaluate_model.py --model_pb_path=deep_v4_sdm/export/${sdm_version} \
  --item_fea_path=${easy_rec_ext_data_dir}/candidate_item_all.txt \
  --eval_sample_path=${easy_rec_ext_data_dir}/sample_v4_eval_5w.txt \
  --vector_dim=50 --topks=10,100,200 \
  --default_user_fea="1,271696|818610|4954999|929177|2278603|3219016|2028434,411153|411153|411153|4801426|3002561|3002561|4801426,2|2|3|4|4|4|4,1|2|3|4|5|6|7,2104483|3219016|2041056|4954999|4198227|1323189|4666650|79715|4954999|3682069|4170517|3911125|1340922|4170517|2087357|3157558|2087357|4973305|46259|3239041|2791761|1305059|4092065|266784|46259|266784|4152983|4615417|3239041|5002615|2734026|5002615|2286574|1338525|3108797|2951368|2266567|1531036|3745169|3827899|230380|4606018|4365585|3830808|2576651|2333346|2268318,4756105|3002561|4801426|411153|1320293|3524510|4756105|2355072|411153|4690421|149192|982926|4690421|149192|2131531|2520771|2131531|2520771|149192|2355072|2355072|2520771|2355072|2520771|149192|2520771|2355072|4145813|2355072|2520377|4145813|2520377|2465336|149192|2355072|1080785|4145813|2920476|2891509|2920476|411153|2735466|2520377|4181361|149192|2520771|2520377,5|5|5|5|5|5|6|6|6|6|6|6|6|7|7|7|7|7|7|7|8|8|8|8|8|8|8|8|8|9|9|9|9|9|10|10|10|10|10|10|11|11|11|11|12|12|12,1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47,-1,-1,-1,-1" \
  --default_item_fea="2041056,4801426"
echo "完成sdm模型评估"