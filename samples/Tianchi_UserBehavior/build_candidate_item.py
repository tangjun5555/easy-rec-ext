# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2022/6/30 9:50
# desc:

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
args = parser.parse_args()
print("Run params:" + str(args))

item_cnt = dict()
all_item_fea_dict = dict()
label_item_fea_dict = dict()

user_id = None
last_behavior = None
line_num = 0
with open(args.input_path, mode="r") as f:
    for line in f:
        line_num += 1
        if line_num % 10000 == 0:
            print(line_num, line)

        split = line.strip().split(",")
        assert len(split) == 5, "错误行:" + str(line_num)

        if split[0] != user_id:
            if user_id:
                label_item_fea_dict[last_behavior[1]] = last_behavior[1] + "," + last_behavior[2]
            user_id = split[0]

        all_item_fea_dict[split[1]] = split[1] + "," + split[2]
        item_cnt[split[1]] = item_cnt.get(split[1], 0) + 1
        last_behavior = split
    label_item_fea_dict[last_behavior[1]] = last_behavior[1] + "," + last_behavior[2]

with open(args.output_path + "_all.txt", mode="w") as fout:
    for k, v in all_item_fea_dict.items():
        fout.write(k + "#" + v + "\n")

with open(args.output_path + "_label.txt", mode="w") as fout:
    for k, v in label_item_fea_dict.items():
        fout.write(k + "#" + v + "\n")

with open(args.output_path + "_hot.txt", mode="w") as fout:
    pairs = list(item_cnt.items())
    pairs.sort(key=lambda x: x[1], reverse=True)
    pairs = [x[0] for x in pairs[:500000]]
    for k in pairs:
        fout.write(k + "#" + all_item_fea_dict[k] + "\n")
