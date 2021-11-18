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
        logging.info("FM, name:%s, field_num:%d, embed_size:%d" % (self.name, field_num, embed_size))

        sum_square = tf.square(tf.reduce_sum(input_value, 1, keepdims=False))
        square_sum = tf.reduce_sum(tf.square(input_value), 1, keepdims=False)
        return 0.5 * tf.subtract(sum_square, square_sum)


class InnerProduct(object):
    def __init__(self, name):
        self.name = name

    def __call__(self, input_value):
        """
        TODO
        Input shape
            - seq_value is a 3D tensor with shape: (batch_size, field_num, embedding_size)

        Output shape
            - 2D tensor with shape: (batch_size, pairs)
        """
        field_num = input_value.get_shape().as_list()[1]
        embed_size = input_value.get_shape().as_list()[2]
        logging.info("InnerProduct, name:%s, field_num:%d, embed_size:%d" % (self.name, field_num, embed_size))


class CAN(object):
    def __init__(self, name, mode, dimension, order, mlp_units=(64, 32)):
        """
        Co-action Network
        Args:
            name:
            mode: str, must be [sequence, non-sequence]
        """
        self.name = name
        self.mode = mode
        self.dimension = dimension
        self.order = order
        self.mlp_units = mlp_units

    def __call__(self, user_value, item_value):
        """
        TODO
        Args:
            user_value:
                3D tensor with shape: (batch_size, user_field_num, embedding_size)
                fed into MLP
            item_value:
                3D tensor with shape: (batch_size, item_field_num, embedding_size)
                server as the weight and bias
        Returns:

        """
        user_value_shape = user_value.get_shape().as_list()
        item_value_shape = item_value.get_shape().as_list()

        assert item_value_shape[-1] == sum(self.mlp_units) + len(self.mlp_units)
