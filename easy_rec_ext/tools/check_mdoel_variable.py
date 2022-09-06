# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2022/7/6 8:01 下午
# desc:

import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, required=True)
parser.add_argument("--variable_name", type=str, required=False, default=None)
args = parser.parse_args()
print("Run params:" + str(args))

checkpoint_path = tf.train.latest_checkpoint(args.checkpoint_path)
reader = tf.train.load_checkpoint(checkpoint_path)

var_to_shape_map = reader.get_variable_to_shape_map()
print("打印变量的shape:")
for key, value in var_to_shape_map.items():
    print(key, value)

if args.variable_name:
    tensor = reader.get_tensor(args.variable_name).as_numpy()
    print("打印变量{variable_name}的值:".format(variable_name=args.variable_name))
    print(tensor)
