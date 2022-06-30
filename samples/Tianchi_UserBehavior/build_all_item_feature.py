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

item_fea_dict = dict()

with open(args.input_path, mode="r") as f:
    line_num = 0
    for line in f:
        line_num += 1
        if line_num % 10000 == 0:
            print("current line num:", line_num)
        split = line.strip().split(",")
        assert len(split) == 5, split
        item_fea_dict[split[1]] = split[1] + "," + split[2]

with open(args.output_path, mode="w") as fout:
    for k, v in item_fea_dict.items():
        fout.write(k + "#" + v + "\n")
