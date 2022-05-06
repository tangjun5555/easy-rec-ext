# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2022/5/4 3:38 PM
# desc:

import os
import logging
import tensorflow as tf
from easy_rec_ext.layers.common_layers import leaky_relu
from easy_rec_ext.utils import variable_util

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


class AutoDis(object):
    def __init__(self, name, in_dim, each_out_size, meta_emb_num=8, temperature=1.0):
        self.name = name
        self.in_dim = in_dim
        self.each_out_size = each_out_size

        self.meta_emb_num = meta_emb_num
        self.temperature = temperature

    def apply_one_dim(self, input_value, sub_name):
        """
        Input shape
            - input_value is a 3D tensor with shape: (batch_size, 1)
        Output shape
            - 2D tensor with shape: (batch_size, each_out_size)
        """
        # Meta-Embeddings
        meta_embedding = variable_util.get_normal_variable(
            scope="AutoDis",
            name=sub_name + "_meta_embeddings",
            shape=(self.meta_emb_num, self.each_out_size),
        )

        # Automatic Discretization
        h = tf.layers.dense(
            input_value,
            self.meta_emb_num,
            use_bias=False,
            name=sub_name + "_mlp1"
        )
        h = leaky_relu(h)
        score = tf.layers.dense(
            h,
            self.meta_emb_num,
            use_bias=False,
            name=sub_name + "_mlp2"
        )
        score = score + h  # skip-connection
        score = score / self.temperature
        score = tf.nn.softmax(score)

        # Aggregation Function
        output = tf.matmul(score, meta_embedding)
        return output

    def __call__(self, input_value):
        """
        Input shape
            - input_value is a 3D tensor with shape: (batch_size, in_dim)
        Output shape
            - 2D tensor with shape: (batch_size, in_dim * each_out_size)
        """
        if self.in_dim == 1:
            return self.apply_one_dim(input_value, self.name)
        else:
            output_list = []
            for i in range(self.in_dim):
                sub_input_value = tf.slice(input_value, [0, i], [-1, 1])
                output_list.append(self.apply_one_dim(sub_input_value, self.name + "_%d" % i))
            return tf.concat(output_list, axis=1)
