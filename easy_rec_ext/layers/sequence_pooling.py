# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/24 6:52 下午
# desc:

import os
import logging
from typing import List

import tensorflow as tf

tf = tf.compat.v1

filename = str(os.path.basename(__file__)).split(".")[0]


class GRUConfig(object):
    def __init__(self, gru_units: List[int], go_backwards: int):
        self.gru_units = gru_units
        self.go_backwards = go_backwards

    @staticmethod
    def handle(data):
        res = GRUConfig(data["gru_units"], data["go_backwards"])
        return res


class SequencePoolingConfig(object):
    def __init__(self, mode: str = "sum", gru_config: GRUConfig = None):
        self.mode = mode
        self.gru_config = gru_config

    @staticmethod
    def handle(data):
        res = SequencePoolingConfig()
        if "mode" in data:
            res.mode = data["mode"]
        if "gru_config" in data:
            res.gru_config = GRUConfig.handle(data["gru_config"])
        return res


class SequencePooling(object):
    """
    The SequencePoolingLayer is used to apply pooling operation
    on variable-length sequence feature/multi-value feature.

    Arguments
        name: str.
        mode: str. Pooling operation to be used, can be sum, mean or max.
        gru_config: str.
    """

    def __init__(self, name, mode="sum", gru_config: GRUConfig = None):
        self.name = name
        self.mode = mode
        self.gru_config = gru_config

    def __call__(self, seq_value, seq_len):
        """
        Input shape
            - seq_value is a 3D tensor with shape: (batch_size, T, embedding_size)
            - seq_len is a 2D tensor with shape : (batch_size), indicate valid length of each sequence

        Output shape
            - 2D tensor with shape: (batch_size, embedding_size)
        """
        seq_len_max = seq_value.get_shape().as_list()[1]
        embedding_size = seq_value.get_shape().as_list()[2]

        seq_len = tf.expand_dims(seq_len, 1)

        mask = tf.sequence_mask(seq_len, maxlen=seq_len_max, dtype=tf.dtypes.float32)
        mask = tf.transpose(mask, perm=(0, 2, 1))
        mask = tf.tile(mask, [1, 1, embedding_size])  # (batch_size, T, embedding_size)

        if self.mode == "max":
            hist = seq_value - (1 - mask) * 1e9
            return tf.reduce_max(hist, axis=1, keep_dims=False)
        elif self.mode == "sum":
            hist = tf.reduce_sum(seq_value * mask, axis=1, keep_dims=False)
            return hist
        elif self.mode == "mean":
            hist = tf.reduce_sum(seq_value * mask, axis=1, keep_dims=False)
            return tf.div(hist, tf.cast(seq_len, tf.float32) + 1e-8)
        elif self.mode == "gru":
            go_backwards = self.gru_config.go_backwards == 1
            gru_input = seq_value
            for i, j in enumerate(self.gru_config.gru_units):
                gru_input, gru_states = tf.keras.layers.GRU(
                    units=j,
                    # stateful=True,
                    return_state=True,
                    go_backwards=go_backwards,
                    name='{}_gru_{}'.format(self.name, str(i)),
                )(gru_input)
                logging.info("%s %s, gru_input.shape:%s, gru_states:%s" % (filename, self.name, str(gru_input.shape), str(gru_states.shape)))
            return tf.reshape(gru_input, (-1, embedding_size))
        else:
            raise ValueError("mode:%s not supported." % self.mode)
