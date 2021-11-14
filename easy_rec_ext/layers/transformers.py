# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/11/9 11:20 下午
# desc:

import logging
import numpy as np
import tensorflow as tf

tf = tf.compat.v1


class PositionEncoding(object):
    def __init__(self,
                 name,
                 pos_embedding_trainable=True,
                 zero_pad=False,
                 scale=True,
                 ):
        self.name = name
        self.pos_embedding_trainable = pos_embedding_trainable
        self.zero_pad = zero_pad
        self.scale = scale

    def __call__(self, inputs):
        _, T, num_units = inputs.get_shape().as_list()

        position_enc = np.array([
            [pos / np.power(10000, 2. * (i // 2) / num_units) for i in range(num_units)]
            for pos in range(T)
        ])
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        lookup_table = tf.get_variable(
            name=self.name + "/" + "lookup_table",
            shape=(T, num_units),
            initializer=tf.initializers.identity(position_enc),
            trainable=self.pos_embedding_trainable,
            dtype=tf.float32
        )
        position_ind = tf.expand_dims(tf.range(T), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)
        if self.scale:
            outputs = outputs * (num_units ** 0.5)
        return outputs + inputs


class MultiHeadAttention(object):
    def __init__(self):
        pass
