# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2023/6/15 19:27
# desc:

import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
args = parser.parse_args()
print("Run params:" + str(args))

train_dts = ["20171130", "20171201", "20171202"]
eval_dt = "20171203"
train_output_files = [open(args.output_path + "_train_%s.txt" % dt, mode="w") for dt in train_dts]
eval_output = open(args.output_path + "_eval_%s.txt" % eval_dt, mode="w")

behavior_type_enum = ["pv", "buy", "cart", "fav"]
max_seq_len = 20


def build_user_sequence_feature(history_behavior_list, need_behavior_type):
    hist_item_id_list = []
    hist_cate_id_list = []
    hist_behavior_type_list = []

    if not history_behavior_list:
        history_behavior_list = []
    for target_behavior in history_behavior_list:
        hist_item_id_list.append(target_behavior[0])
        hist_cate_id_list.append(target_behavior[1])
        if need_behavior_type:
            hist_behavior_type_list.append(str(behavior_type_enum.index(target_behavior[2]) + 1))

    while len(hist_item_id_list) < max_seq_len:
        hist_item_id_list.append(str(-1))
        hist_cate_id_list.append(str(-1))
        if need_behavior_type:
            hist_behavior_type_list.append(str(-1))

    if need_behavior_type:
        return ",".join([
            "|".join(hist_item_id_list),
            "|".join(hist_cate_id_list),
            "|".join(hist_behavior_type_list),
        ])
    else:
        return ",".join([
            "|".join(hist_item_id_list),
            "|".join(hist_cate_id_list),
        ])


def build_user_feature(user_id, history_behavior_list):
    if not history_behavior_list:
        history_behavior_list = []

    target_history_behavior_list = [target_behavior for target_behavior in history_behavior_list
                                    if target_behavior[2] == "pv"]
    target_history_behavior_list = target_history_behavior_list[:max_seq_len]
    user_clk_sequence_feature = build_user_sequence_feature(target_history_behavior_list, False)

    target_history_behavior_list = [target_behavior for target_behavior in history_behavior_list if
                                    target_behavior[2] != "pv"]
    target_history_behavior_list = target_history_behavior_list[:max_seq_len]
    user_buy_sequence_feature = build_user_sequence_feature(target_history_behavior_list, True)

    return ",".join([
        user_id,
        user_clk_sequence_feature,
        user_buy_sequence_feature,
    ])


def build_item_feature(current_behavior):
    return current_behavior[0] + "," + current_behavior[1]


user_id = None
user_seq_list = []
line_num = 0
with open(args.input_path, mode="r") as f:
    for line in f:
        line_num += 1
        if line_num % 50000 == 0:
            print(line_num, line)
        split = line.strip().split(",")
        assert len(split) == 5, "错误行:" + str(line_num)

        if split[0] != user_id or line_num == 100150807:
            if user_id:
                for i, target_behavior in enumerate(user_seq_list):
                    current_dt = time.strftime("%Y%m%d", time.localtime(int(target_behavior[3])))
                    if (current_dt in train_dts or current_dt == eval_dt) and int(user_id) % 10 == 5:
                        history_behavior_list = user_seq_list[:i]
                        user_feature = build_user_feature(user_id, history_behavior_list[::-1])
                    if current_dt in train_dts and int(user_id) % 10 == 5:
                        train_output_files[train_dts.index(current_dt)].write(
                            ",".join([
                                str(1),
                                user_feature,
                                build_item_feature(target_behavior),
                            ])
                            + "\n"
                        )
                    elif current_dt == eval_dt and int(user_id) % 10 == 5:
                        eval_output.write("\t".join([
                            user_feature,
                            target_behavior[0],
                        ]) + "\n")

            user_id = split[0]
            user_seq_list = []

        current_behavior = split[1:]
        if user_seq_list:
            assert int(current_behavior[3]) >= int(user_seq_list[-1][3]), "错误行:" + str(line_num)
        user_seq_list.append(current_behavior)

for train_output in train_output_files:
    train_output.close()
eval_output.close()
