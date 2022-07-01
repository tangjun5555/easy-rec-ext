# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2022/6/26 14:50
# desc:

import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--output_suffix", type=str, required=False, default=None)
parser.add_argument("--seq_max_length", type=int, required=False, default=50)
args = parser.parse_args()
print("Run params:" + str(args))

behavior_type_enum = ["pv", "buy", "cart", "fav"]


def build_user_feature(user_id, current_behavior, history_behavior_list):
    hist_item_ids = []
    hist_item_cate_ids = []
    hist_behavior_type_list = []
    hist_behavior_time_diff_list = []
    hist_behavior_time_rank_list = []
    if not history_behavior_list:
        history_behavior_list = []
    history_behavior_list = history_behavior_list[:args.seq_max_length]

    for target_behavior in history_behavior_list:
        hist_item_ids.append(target_behavior[0])
        hist_item_cate_ids.append(target_behavior[1])
        hist_behavior_type_list.append(str(behavior_type_enum.index(target_behavior[2]) + 1))

        diff = int(current_behavior[3]) - int(target_behavior[3])
        assert diff >= 0, str(current_behavior) + str(target_behavior)
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
    # while len(hist_item_ids) < args.seq_max_length:
    #     hist_item_ids.append(str(-1))
    #     hist_item_cate_ids.append(str(-1))
    #     hist_behavior_type_list.append(str(-1))
    #     hist_behavior_time_diff_list.append(str(-1))
    #     hist_behavior_time_rank_list.append(str(len(hist_behavior_time_rank_list) + 1))
    return ",".join([
        user_id,
        "|".join(hist_item_ids),
        "|".join(hist_item_cate_ids),
        "|".join(hist_behavior_type_list),
        "|".join(hist_behavior_time_diff_list),
        "|".join(hist_behavior_time_rank_list),
    ])


def build_item_feature(current_behavior):
    return current_behavior[0] + "," + current_behavior[1]


dts = [
    # "20171125", "20171126", "20171127", "20171128",
    # "20171129", "20171130",
    "20171201", "20171202", "20171203"
]
if args.output_suffix:
    dt_output_list = [open("%s/sample_%s_%s.csv" % (args.output_dir, args.output_suffix, dt), mode="w") for dt in dts]
else:
    dt_output_list = [open("%s/sample_%s.csv" % (args.output_dir, dt), mode="w") for dt in dts]
user_id = None
user_seq_list = []
with open(args.input_path, mode="r") as f:
    line_num = 0
    for line in f:
        line_num += 1
        split = line.strip().split(",")
        assert len(split) == 5, split

        if split[0] != user_id:
            if user_id and int(user_id) % 1000 == 0:
                print("line_num:", line_num)
                print("change user", user_id, user_seq_list)
            user_id = split[0]
            user_seq_list = []

        current_behavior = split[1:]
        if user_seq_list:
            assert int(current_behavior[3]) >= int(user_seq_list[0][3]), str(current_behavior) + str(user_seq_list[0])
        try:
            current_dt = time.strftime("%Y%m%d", time.localtime(int(current_behavior[3])))
        except Exception as e:
            print("【Error】")
            print("【Error】", current_behavior)
            print("【Error】")
            time.sleep(2)
            continue

        if current_dt not in dts:
            print("current_dt is out of range", current_dt, current_behavior)
        else:
            dt_output_list[dts.index(current_dt)] \
                .write(
                ",".join([str(1),
                          build_user_feature(user_id, current_behavior, user_seq_list),
                          build_item_feature(current_behavior),
                          ])
                + "\n"
            )
        user_seq_list.insert(0, current_behavior)

for i in range(len(dts)):
    dt_output_list[i].close()
