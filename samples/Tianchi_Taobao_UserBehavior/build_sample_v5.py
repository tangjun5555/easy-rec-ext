# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2022/6/26 14:50
# desc:

import time
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--all_item_path", type=str, required=True)
parser.add_argument("--user_train_neg_num", type=int, required=False, default=10)
parser.add_argument("--st_seq_max_length", type=int, required=False, default=20)
parser.add_argument("--lt_seq_max_length", type=int, required=False, default=50)
parser.add_argument("--conversion_seq_max_length", type=int, required=False, default=10)
args = parser.parse_args()
print("Run params:" + str(args))

behavior_type_enum = ["pv", "buy", "cart", "fav"]
dividing_time_point = 24 * 60 * 60

# train_dts = ["20171126", "20171127", "20171128", "20171129", "20171130", "20171201", "20171202"]
train_dts = ["20171130", "20171201", "20171202"]
eval_dt = "20171203"
train_output_files = [open(args.output_path + "_train_%s.txt" % dt, mode="w") for dt in train_dts]
eval_output = open(args.output_path + "_eval.txt", mode="w")


def build_user_seq_feature(current_behavior, history_behavior_list):
    hist_item_ids = []
    hist_item_cate_ids = []
    hist_behavior_time_diff_list = []
    hist_behavior_time_rank_list = []

    if not history_behavior_list:
        history_behavior_list = []

    for target_behavior in history_behavior_list:
        hist_item_ids.append(target_behavior[0])
        hist_item_cate_ids.append(target_behavior[1])

        diff = int(current_behavior[3]) - int(target_behavior[3])
        if diff <= (30 * 60):
            value = 0
        elif diff <= (4 * 60 * 60):
            value = 1
        elif diff <= (8 * 60 * 60):
            value = 2
        elif diff <= (1 * 24 * 60 * 60):
            value = 3
        elif diff <= (2 * 24 * 60 * 60):
            value = 4
        elif diff <= (3 * 24 * 60 * 60):
            value = 5
        elif diff <= (4 * 24 * 60 * 60):
            value = 6
        elif diff <= (5 * 24 * 60 * 60):
            value = 7
        elif diff <= (6 * 24 * 60 * 60):
            value = 8
        elif diff <= (7 * 24 * 60 * 60):
            value = 9
        elif diff <= (8 * 24 * 60 * 60):
            value = 10
        elif diff <= (9 * 24 * 60 * 60):
            value = 11
        else:
            value = 12
        hist_behavior_time_diff_list.append(str(value + 1))
        hist_behavior_time_rank_list.append(str(len(hist_behavior_time_rank_list) + 1))

    if not hist_item_ids:
        hist_item_ids = [str(-1)]
        hist_item_cate_ids = [str(-1)]
        hist_behavior_time_diff_list = [str(-1)]
        hist_behavior_time_rank_list = [str(-1)]
    return ",".join([
        "|".join(hist_item_ids),
        "|".join(hist_item_cate_ids),
        # "|".join(hist_behavior_time_diff_list),
        "|".join(hist_behavior_time_rank_list),
    ])


def build_user_feature(user_id, current_behavior, history_behavior_list):
    if not history_behavior_list:
        history_behavior_list = []

    target_history_behavior_list = [target_behavior for target_behavior in history_behavior_list
                                    if target_behavior[2] == "pv"
                                    and (int(current_behavior[3]) - int(target_behavior[3])) <= dividing_time_point]
    target_history_behavior_list = target_history_behavior_list[:args.st_seq_max_length]
    user_short_term_seq_feature = build_user_seq_feature(current_behavior, target_history_behavior_list)

    target_history_behavior_list = [target_behavior for target_behavior in history_behavior_list
                                    if target_behavior[2] == "pv"
                                    and (int(current_behavior[3]) - int(target_behavior[3])) > dividing_time_point]
    target_history_behavior_list = target_history_behavior_list[:args.lt_seq_max_length]
    user_long_term_seq_feature = build_user_seq_feature(current_behavior, target_history_behavior_list)

    target_history_behavior_list = [target_behavior for target_behavior in history_behavior_list
                                    if target_behavior[2] != "pv"]
    target_history_behavior_list = target_history_behavior_list[:args.conversion_seq_max_length]
    user_conversion_seq_feature = build_user_seq_feature(current_behavior, target_history_behavior_list)

    return ",".join([
        user_id,
        user_short_term_seq_feature,
        user_long_term_seq_feature,
        user_conversion_seq_feature,
    ])


def build_item_feature(current_behavior):
    return current_behavior[0] + "," + current_behavior[1]


def load_all_item_fea(input_file):
    all_item_fea_dict = dict()
    with open(input_file, mode="r") as f:
        for line in f:
            split = line.strip().split("#")
            all_item_fea_dict[split[0]] = split[1]
    return all_item_fea_dict


def global_neg_sample(all_ids, num, remove_ids=None):
    if not remove_ids:
        remove_ids = []
    res = []
    try_cnt = 0
    while try_cnt < (num + 10):
        cur_id = all_ids[random.randint(0, len(all_ids) - 1)]
        if cur_id in res or cur_id in remove_ids:
            continue
        else:
            res.append(cur_id)
        if len(res) >= num:
            break
        try_cnt += 1
    return res


all_item_fea_dict = load_all_item_fea(args.all_item_path)
all_item_ids = list(all_item_fea_dict.keys())

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
                    history_behavior_list = user_seq_list[:i]
                    hist_item_ids = [x[0] for x in history_behavior_list]
                    user_feature = build_user_feature(user_id, target_behavior, history_behavior_list[::-1])

                    if current_dt in train_dts:
                        train_output_files[train_dts.index(current_dt)].write(
                            ",".join([
                                str(1),
                                user_feature,
                                build_item_feature(target_behavior),
                            ])
                            + "\n"
                        )
                        neg_ids = global_neg_sample(all_item_ids, args.user_train_neg_num, hist_item_ids)
                        for neg_id in neg_ids:
                            train_output_files[train_dts.index(current_dt)].write(
                                ",".join([
                                    str(0),
                                    user_feature,
                                    all_item_fea_dict[neg_id],
                                ])
                                + "\n"
                            )
                    elif current_dt == eval_dt and int(user_id) % 10 == 5:
                        # eval_output.write("#".join([
                        #     user_feature,
                        #     ",".join(hist_item_ids[i:])
                        # ]) + "\n")
                        # break
                        eval_output.write("#".join([
                            user_feature,
                            target_behavior[0]
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
