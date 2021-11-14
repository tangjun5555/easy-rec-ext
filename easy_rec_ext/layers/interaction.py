# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/11 6:32 下午
# desc:

import os
import logging

import tensorflow as tf

tf = tf.compat.v1

filename = str(os.path.basename(__file__)).split(".")[0]


class InteractionConfig(object):
    def __init__(self, mode):
        self.mode = mode

    @staticmethod
    def handle(data):
        res = InteractionConfig(data["mode"])
        return res


class FM(object):
    def __init__(self, name):
        self.name = name

    def __call__(self, input_value):
        """
        Input shape
            - seq_value is a 3D tensor with shape: (batch_size, field_num, embedding_size)

        Output shape
            - 2D tensor with shape: (batch_size, embedding_size)
        """
        field_num = input_value.get_shape().as_list()[1]
        embed_size = input_value.get_shape().as_list()[2]
        logging.info("fm, name:%s, field_num:%d, embed_size:%d" % (self.name, field_num, embed_size))

        sum_square = tf.square(tf.reduce_sum(input_value, 1, keep_dims=False))
        square_sum = tf.reduce_sum(tf.square(input_value), 1, keep_dims=False)
        return 0.5 * tf.subtract(sum_square, square_sum)


class CAN(object):
    def __init__(self, name, ):
        self.name = name

    def __call__(self, input_value):
        pass
