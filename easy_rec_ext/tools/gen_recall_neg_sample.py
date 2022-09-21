# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2022/9/6 17:19
# desc:

import argparse
import tensorflow as tf
from easy_rec_ext.utils import sampler_ops

if tf.__version__ >= "2.0":
    tf = tf.compat.v1

parser = argparse.ArgumentParser()
parser.add_argument("--sampler_type", type=str, required=True,
                    help="负采样类型",
                    choices=[
                        "negative_sampler",  # 加权随机负采样，会排除Mini-Batch内的ItemId
                        "negative_sampler_v2",  # 加权随机负采样，会跟排除Mini-Batch内的User有边的Item Id
                        "hard_negative_sampler",  # 加权随机负采样，会排除Mini-Batch内的Item Id，同时HardNegative边表中(一般为曝光未点击)进行负采样作为HardNegative
                        "hard_negative_sampler_v2",  # 加权随机负采样，会跟排除Mini-Batch内的User有边的Item Id，同时HardNegative边表中(一般为曝光未点击)进行负采样作为HardNegative
                    ],
                    )
parser.add_argument("--sampling_type", type=str, required=False, default="UniformSampling",
                    help="负采样随机类型",
                    choices=[
                        "UniformSampling",
                        "WeightedSampling",
                        "W2VSampling",
                    ],
                    )
parser.add_argument("--item_input_path", type=str, required=True,
                    help="Item表，Schema为: itemid:string\tweight:float\titem_attrs:string",
                    )
parser.add_argument("--pos_input_path", type=str, required=True,
                    help="正样本表，Schema为: label:int,features",
                    )
parser.add_argument("--pos_edge_input_path", type=str, required=False, default=None,
                    help="Positive边表，Schema为: userid:string\titemids:list_string",
                    )
parser.add_argument("--hard_neg_edge_input_path", type=str, required=False, default=None,
                    help="HardNegative边表，Schema为: userid:string\titemid:list_string",
                    )
parser.add_argument("--item_fea_num", type=int, required=True, help="物品特征数量")
parser.add_argument("--easy_neg_num", type=int, required=False, default=100, help="easy负样本数量")
parser.add_argument("--hard_neg_num", type=int, required=False, default=10, help="hard负样本数量")
args = parser.parse_args()
print("Run params:" + str(args))

if args.sampler_type == "negative_sampler":
    sampler = sampler_ops.NegativeSampler(
        args.item_input_path,
        sampling_type=args.sampling_type,
    )
elif args.sampler_type == "negative_sampler_v2":
    assert args.pos_edge_input_path
    sampler = sampler_ops.NegativeSamplerV2(
        args.item_input_path,
        pos_edge_input_path=args.pos_edge_input_path,
        sampling_type=args.sampling_type,
    )
elif args.sampler_type == "hard_negative_sampler":
    assert args.hard_neg_edge_input_path
    sampler = sampler_ops.HardNegativeSampler(
        args.item_input_path,
        hard_neg_edge_input_path=args.hard_neg_edge_input_path,
        num_hard_sample=args.num_hard_sample,
        sampling_type=args.sampling_type,
    )
else:
    assert args.pos_edge_input_path
    assert args.hard_neg_edge_input_path
    sampler = sampler_ops.HardNegativeSamplerV2(
        args.item_input_path,
        pos_edge_input_path=args.pos_edge_input_path,
        hard_neg_edge_input_path=args.hard_neg_edge_input_path,
        num_hard_sample=args.num_hard_sample,
        sampling_type=args.sampling_type,
    )

fin = open(args.pos_input_path, mode="r")
fout = open(args.pos_input_path + "_train", mode="w")

line_num = 0
for line in fin:
    line_num += 1
    fout.write(line)

    split = line.strip().split(",")
    batch_item_ids = [split[-args.item_fea_num]]
    uid = split[1]

    neg_samples = sampler.get_samples(uid, args.easy_neg_num, batch_item_ids)
    for one_sample in neg_samples:
        fout.write("0," + ",".join(split[1:-args.item_fea_num]) + "," + one_sample + "\n")

    if line_num % 10000 == 0:
        print(line)

fout.close()
fin.close()
