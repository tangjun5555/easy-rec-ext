# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2022/3/26 7:29 PM
# desc:

import os
import logging
import tensorflow as tf
from easy_rec_ext.utils import variable_util

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


class SENetLayer(object):
    """
    squeeze_excitation_layer
    """
    def __init__(self, name, reduction_ratio=2):
        self.name = name
        self.reduction_ratio = reduction_ratio

    def __call__(self, input_value):
        """
        Input shape
            - input_value is a 3D tensor with shape: (batch_size, field_num, embedding_size)

        Output shape
            - output_value is a 3D tensor with shape: (batch_size, field_num, embedding_size)
        """
        field_num = input_value.get_shape().as_list()[1]
        embed_size = input_value.get_shape().as_list()[2]
        reduction_size = max(1, field_num // self.reduction_ratio)
        logging.info("SENetLayer, name:%s, field_num:%d, embed_size:%d, reduction_size:%d"
                     % (self.name, field_num, embed_size, reduction_size))

        W_1 = variable_util.get_normal_variable(
            scope="SENetLayer", name=self.name + "_W1", shape=(field_num, reduction_size)
        )
        W_2 = variable_util.get_normal_variable(
            scope="SENetLayer", name=self.name + "_W2", shape=(reduction_size, field_num)
        )

        # Squeeze Step
        Z = tf.math.reduce_mean(input_value, axis=-1, keepdims=False)
        logging.info("SENetLayer, name:%s, Z.shape:%s" % (self.name, str(Z.shape)))
        # Excitation Step
        A_1 = tf.nn.relu(tf.tensordot(Z, W_1, axes=(-1, 0)))
        logging.info("SENetLayer, name:%s, A_1.shape:%s" % (self.name, str(A_1.shape)))
        A_2 = tf.nn.relu(tf.tensordot(A_1, W_2, axes=(-1, 0)))
        logging.info("SENetLayer, name:%s, A_2.shape:%s" % (self.name, str(A_2.shape)))
        # Re-Weight Step
        return tf.multiply(input_value, tf.expand_dims(A_2, axis=2))
