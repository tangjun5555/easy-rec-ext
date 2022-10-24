# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2022/3/26 7:29 PM
# desc:

import os
import logging
import tensorflow as tf
from easy_rec_ext.utils import variable_util
from easy_rec_ext.layers import layer_norm

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


class SENetLayer(object):
    """
    Squeeze-and-Excitation Network Layer
    """
    def __init__(self, name,
                 reduction_ratio=1.1,
                 squeeze_fun="mean",
                 squeeze_group_num=1,
                 use_skip_connection=False,
                 use_layer_norm=False,
                 ):
        self.name = name
        self.reduction_ratio = reduction_ratio
        assert squeeze_fun in ["mean", "max"]
        self.squeeze_fun = squeeze_fun
        self.squeeze_group_num = squeeze_group_num
        self.use_skip_connection = use_skip_connection
        self.use_layer_norm = use_layer_norm

    def __call__(self, input_value):
        """
        Input shape
            - input_value is a 3D tensor with shape: (batch_size, field_num, embedding_size)

        Output shape
            - output_value is a 3D tensor with shape: (batch_size, field_num, embedding_size)
        """
        field_num = input_value.get_shape().as_list()[1]
        embed_size = input_value.get_shape().as_list()[2]
        reduction_size = max(1, int(field_num / self.reduction_ratio))
        logging.info("SENetLayer, name:%s, field_num:%d, embed_size:%d, reduction_size:%d"
                     % (self.name, field_num, embed_size, reduction_size))

        input_dim = output_dim = field_num * self.squeeze_group_num

        W_1 = variable_util.get_normal_variable(
            scope="SENetLayer", name=self.name + "_W1", shape=(input_dim, reduction_size)
        )
        W_2 = variable_util.get_normal_variable(
            scope="SENetLayer", name=self.name + "_W2", shape=(reduction_size, output_dim)
        )

        # Squeeze Step
        Z = tf.reshape(input_value, (-1, field_num * self.squeeze_group_num, embed_size // self.squeeze_group_num))
        if self.squeeze_fun == "mean":
            Z = tf.math.reduce_mean(Z, axis=-1, keepdims=False)
        else:
            Z = tf.math.reduce_max(Z, axis=-1, keepdims=False)
        logging.info("SENetLayer, name:%s, Z.shape:%s" % (self.name, str(Z.shape)))

        # Excitation Step
        A_1 = tf.nn.relu(tf.tensordot(Z, W_1, axes=(-1, 0)))
        logging.info("SENetLayer, name:%s, A_1.shape:%s" % (self.name, str(A_1.shape)))
        A_2 = tf.nn.relu(tf.tensordot(A_1, W_2, axes=(-1, 0)))
        logging.info("SENetLayer, name:%s, A_2.shape:%s" % (self.name, str(A_2.shape)))

        # Re-Weight Step
        res = tf.reshape(input_value, (-1, field_num * self.squeeze_group_num, embed_size // self.squeeze_group_num))
        res = tf.multiply(res, tf.expand_dims(A_2, axis=2))
        res = tf.reshape(res, (-1, field_num * embed_size))

        if self.use_skip_connection:
            res = tf.math.add(res, input_value)
        if self.use_layer_norm:
            res = layer_norm.LayerNormalization(
                res.get_shape().as_list()[-1],
                self.name + "_LN",
            )(res)
        return res
