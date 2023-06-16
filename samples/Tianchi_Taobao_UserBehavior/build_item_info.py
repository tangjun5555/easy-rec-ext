# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2023/6/16 16:34
# desc:

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--frequency_as_weight", type=int, required=False, default=0)
args = parser.parse_args()
print("Run params:" + str(args))

item_feature_dict = dict()
item_frequency_dict = dict()


def build_item_feature(current_behavior):
    return current_behavior[0] + ":" + current_behavior[1]


line_num = 0
with open(args.input_path, mode="r") as fin:
    for line in fin:
        line_num += 1
        if line_num % 50000 == 0:
            print(line_num, line)
        split = line.strip().split(",")
        assert len(split) == 5, "错误行:" + str(line_num)

        current_behavior = split[1:]
        item_id = current_behavior[0]
        item_feature_dict[item_id] = build_item_feature(current_behavior)
        item_frequency_dict[item_id] = item_frequency_dict.get(item_id, 0) + 1


with open(args.output_path, mode="w") as fout:
    fout.write("\t".join(["id:int64", "weight:float", "feature:string"]) + "\n")
    item_feature_list = sorted(item_feature_dict.items())
    for pair in item_feature_list:
        if args.frequency_as_weight:
            fout.write("\t".join([pair[0], str(item_frequency_dict[pair[0]]), pair[1]]) + "\n")
        else:
            fout.write("\t".join([pair[0], str(1), pair[1]]) + "\n")
