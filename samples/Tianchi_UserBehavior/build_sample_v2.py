# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2022/6/26 14:50
# desc:

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--st_seq_max_length", type=int, required=False, default=10)
parser.add_argument("--lt_seq_max_length", type=int, required=False, default=50)
args = parser.parse_args()
print("Run params:" + str(args))

user_train_num = 10
behavior_type_enum = ["pv", "buy", "cart", "fav"]


def build_user_short_term_seq_feature(current_behavior, history_behavior_list):
    hist_item_ids = []
    hist_item_cate_ids = []
    hist_behavior_type_list = []
    hist_behavior_time_rank_list = []

    if not history_behavior_list:
        history_behavior_list = []
    history_behavior_list = [target_behavior for target_behavior in history_behavior_list if (int(current_behavior[3]) - int(target_behavior[3])) <= 24 * 60 * 60]
    history_behavior_list = history_behavior_list[:args.st_seq_max_length]

    for target_behavior in history_behavior_list:
        hist_item_ids.append(target_behavior[0])
        hist_item_cate_ids.append(target_behavior[1])
        hist_behavior_type_list.append(str(behavior_type_enum.index(target_behavior[2]) + 1))
        hist_behavior_time_rank_list.append(str(len(hist_behavior_time_rank_list) + 1))

    if not hist_item_ids:
        hist_item_ids = [str(-1)]
        hist_item_cate_ids = [str(-1)]
        hist_behavior_type_list = [str(-1)]
        hist_behavior_time_rank_list = [str(-1)]
    return ",".join([
        "|".join(hist_item_ids),
        "|".join(hist_item_cate_ids),
        "|".join(hist_behavior_type_list),
        "|".join(hist_behavior_time_rank_list),
    ])


def build_user_long_term_seq_feature(current_behavior, history_behavior_list):
    hist_item_ids = []
    hist_item_cate_ids = []
    hist_behavior_type_list = []
    hist_behavior_time_diff_list = []
    hist_behavior_time_rank_list = []

    if not history_behavior_list:
        history_behavior_list = []
    history_behavior_list = [target_behavior for target_behavior in history_behavior_list if (int(current_behavior[3]) - int(target_behavior[3])) > 24 * 60 * 60]
    history_behavior_list = history_behavior_list[:args.lt_seq_max_length]

    for target_behavior in history_behavior_list:
        hist_item_ids.append(target_behavior[0])
        hist_item_cate_ids.append(target_behavior[1])
        hist_behavior_type_list.append(str(behavior_type_enum.index(target_behavior[2]) + 1))

        diff = int(current_behavior[3]) - int(target_behavior[3])
        if diff <= (30 * 60):
            value = 0
        elif diff <= (1 * 24 * 60 * 60):
            value = 1
        elif diff <= (2 * 24 * 60 * 60):
            value = 2
        elif diff <= (3 * 24 * 60 * 60):
            value = 3
        elif diff <= (4 * 24 * 60 * 60):
            value = 4
        elif diff <= (5 * 24 * 60 * 60):
            value = 5
        elif diff <= (6 * 24 * 60 * 60):
            value = 6
        elif diff <= (7 * 24 * 60 * 60):
            value = 7
        elif diff <= (8 * 24 * 60 * 60):
            value = 8
        elif diff <= (9 * 24 * 60 * 60):
            value = 9
        else:
            value = 10
        hist_behavior_time_diff_list.append(str(value + 1))
        hist_behavior_time_rank_list.append(str(len(hist_behavior_time_rank_list) + 1))

    if not hist_item_ids:
        hist_item_ids = [str(-1)]
        hist_item_cate_ids = [str(-1)]
        hist_behavior_type_list = [str(-1)]
        hist_behavior_time_diff_list = [str(-1)]
        hist_behavior_time_rank_list = [str(-1)]
    return ",".join([
        "|".join(hist_item_ids),
        "|".join(hist_item_cate_ids),
        "|".join(hist_behavior_type_list),
        "|".join(hist_behavior_time_diff_list),
        "|".join(hist_behavior_time_rank_list),
    ])


def build_user_feature(user_id, current_behavior, history_behavior_list):
    return ",".join([
        user_id,
        build_user_short_term_seq_feature(current_behavior, history_behavior_list),
        build_user_long_term_seq_feature(current_behavior, history_behavior_list),
    ])


def build_item_feature(current_behavior):
    return current_behavior[0] + "," + current_behavior[1]


train_output = open(args.output_path + "_train.txt", mode="w")
eval_output = open(args.output_path + "_eval.txt", mode="w")

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

        if split[0] != user_id:
            if user_id:
                if len(user_seq_list) > 1:
                    history_behavior_list = user_seq_list[1:]
                else:
                    history_behavior_list = []
                eval_output.write(
                    ",".join([
                        str(1),
                        build_user_feature(user_id, user_seq_list[0], history_behavior_list),
                        build_item_feature(user_seq_list[0]),
                    ])
                    + "\n"
                )

                for i in range(1, 1 + user_train_num):
                    if len(user_seq_list) <= i:
                        break
                    if len(user_seq_list) > (i+1):
                        history_behavior_list = user_seq_list[(i+1):]
                    else:
                        history_behavior_list = []
                    train_output.write(
                        ",".join([
                            str(1),
                            build_user_feature(user_id, user_seq_list[i], history_behavior_list),
                            build_item_feature(user_seq_list[i]),
                        ])
                        + "\n"
                    )

            user_id = split[0]
            user_seq_list = []

        current_behavior = split[1:]
        if user_seq_list:
            assert int(current_behavior[3]) >= int(user_seq_list[0][3]), "错误行:" + str(line_num)
        user_seq_list.insert(0, current_behavior)

train_output.close()
eval_output.close()
