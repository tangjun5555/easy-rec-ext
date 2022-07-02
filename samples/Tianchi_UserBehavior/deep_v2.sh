#!/bin/bash

if [ -z ${easy_rec_ext_data_dir} ]; then
  echo "easy_rec_ext_data_dir is not exists"
  exit 1
else
  echo "easy_rec_ext_data_dir=${easy_rec_ext_data_dir}"
fi

python build_candidate_item.py --input_path=${easy_rec_ext_data_dir}/UserBehavior.csv --output_path=${easy_rec_ext_data_dir}/candidate_item
echo "完成构建候选物品特征"

python build_sample_v2.py --input_path=${easy_rec_ext_data_dir}/UserBehavior.csv --output_path=${easy_rec_ext_data_dir}/sample_v2 --st_seq_max_length=10 --lt_seq_max_length=50
echo "完成构建训练样本"

tail -n 10000 ${easy_rec_ext_data_dir}/sample_v2_eval.txt > ${easy_rec_ext_data_dir}/sample_v2_eval_1w.txt
python -m easy_rec_ext.main --pipeline_config_path=deep_v2.json --task_type=train_and_evaluate \
  --train_input_path=${easy_rec_ext_data_dir}/sample_v2_train.txt \
  --eval_input_path=${easy_rec_ext_data_dir}/sample_v2_eval_1w.txt
echo "完成模型训练"

python -m easy_rec_ext.main --pipeline_config_path=deep_v2.json --task_type=export
echo "完成模型导出"

version=`ls deep_v2/export | tail -n 1`
python evaluate_model.py --model_pb_path=deep_v2/export/${version} \
  --item_fea_path=${easy_rec_ext_data_dir}/candidate_item_label.txt \
  --eval_sample_path=${easy_rec_ext_data_dir}/sample_v2_eval.txt \
  --vector_dim=100 --topks=5,20,100 \
  --default_user_fea="1,4954999|4198227|1323189,411153|1320293|3524510,1|1|1,1|2|3,4666650|79715|4954999|3682069|4170517|3911125|1340922|4170517|2087357|3157558|2087357|4973305|46259|3239041|2791761|1305059|4092065|266784|46259|266784|4152983|4615417|3239041|5002615|2734026|5002615|2286574|1338525|3108797|2951368|2266567|1531036|3745169|3827899|230380|4606018|4365585|3830808|2576651|2333346|2268318,4756105|2355072|411153|4690421|149192|982926|4690421|149192|2131531|2520771|2131531|2520771|149192|2355072|2355072|2520771|2355072|2520771|149192|2520771|2355072|4145813|2355072|2520377|4145813|2520377|2465336|149192|2355072|1080785|4145813|2920476|2891509|2920476|411153|2735466|2520377|4181361|149192|2520771|2520377,1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1,3|3|3|3|3|3|3|4|4|4|4|4|4|4|5|5|5|5|5|5|5|5|5|5|5|6|6|6|6|7|7|7|7|7|8|8|8|8|9|9|9,1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41" \
  --default_item_fea="2041056,4801426"
python evaluate_model.py --model_pb_path=deep_v2/export/${version} \
  --item_fea_path=${easy_rec_ext_data_dir}/candidate_item_hot.txt \
  --eval_sample_path=${easy_rec_ext_data_dir}/sample_v2_eval_1w.txt \
  --vector_dim=100 --topks=5,20,100 \
  --default_user_fea="1,4954999|4198227|1323189,411153|1320293|3524510,1|1|1,1|2|3,4666650|79715|4954999|3682069|4170517|3911125|1340922|4170517|2087357|3157558|2087357|4973305|46259|3239041|2791761|1305059|4092065|266784|46259|266784|4152983|4615417|3239041|5002615|2734026|5002615|2286574|1338525|3108797|2951368|2266567|1531036|3745169|3827899|230380|4606018|4365585|3830808|2576651|2333346|2268318,4756105|2355072|411153|4690421|149192|982926|4690421|149192|2131531|2520771|2131531|2520771|149192|2355072|2355072|2520771|2355072|2520771|149192|2520771|2355072|4145813|2355072|2520377|4145813|2520377|2465336|149192|2355072|1080785|4145813|2920476|2891509|2920476|411153|2735466|2520377|4181361|149192|2520771|2520377,1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1,3|3|3|3|3|3|3|4|4|4|4|4|4|4|5|5|5|5|5|5|5|5|5|5|5|6|6|6|6|7|7|7|7|7|8|8|8|8|9|9|9,1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41" \
  --default_item_fea="2041056,4801426"
python evaluate_model.py --model_pb_path=deep_v2/export/${version} \
  --item_fea_path=${easy_rec_ext_data_dir}/candidate_item_all.txt \
  --eval_sample_path=${easy_rec_ext_data_dir}/sample_v2_eval.txt \
  --vector_dim=100 --topks=5,20,100 \
  --default_user_fea="1,4954999|4198227|1323189,411153|1320293|3524510,1|1|1,1|2|3,4666650|79715|4954999|3682069|4170517|3911125|1340922|4170517|2087357|3157558|2087357|4973305|46259|3239041|2791761|1305059|4092065|266784|46259|266784|4152983|4615417|3239041|5002615|2734026|5002615|2286574|1338525|3108797|2951368|2266567|1531036|3745169|3827899|230380|4606018|4365585|3830808|2576651|2333346|2268318,4756105|2355072|411153|4690421|149192|982926|4690421|149192|2131531|2520771|2131531|2520771|149192|2355072|2355072|2520771|2355072|2520771|149192|2520771|2355072|4145813|2355072|2520377|4145813|2520377|2465336|149192|2355072|1080785|4145813|2920476|2891509|2920476|411153|2735466|2520377|4181361|149192|2520771|2520377,1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1,3|3|3|3|3|3|3|4|4|4|4|4|4|4|5|5|5|5|5|5|5|5|5|5|5|6|6|6|6|7|7|7|7|7|8|8|8|8|9|9|9,1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41" \
  --default_item_fea="2041056,4801426"
echo "完成模型评估"
