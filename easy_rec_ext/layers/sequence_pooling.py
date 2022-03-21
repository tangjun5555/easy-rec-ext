# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/24 6:52 下午
# desc:

import os
import logging
from typing import List
import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


class GRUConfig(object):
    def __init__(self, gru_units: List[int], go_backwards: bool):
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
        mode: str. Pooling operation to be used.
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
        if self.mode == "max":
            return tf.reduce_max(seq_value, axis=1, keep_dims=False)
        elif self.mode == "sum":
            return tf.reduce_sum(seq_value, axis=1, keep_dims=False)
        elif self.mode == "mean":
            hist = tf.reduce_sum(seq_value, axis=1, keep_dims=False)
            return tf.div(hist, tf.cast(seq_len, tf.float32) + 1e-8)
        elif self.mode == "gru":
            gru_input = seq_value
            for i, j in enumerate(self.gru_config.gru_units):
                gru_input, gru_states = tf.keras.layers.GRU(
                    units=j,
                    return_state=True,
                    go_backwards=self.gru_config.go_backwards,
                    name="{}_gru_{}".format(self.name, str(i)),
                )(gru_input)
                logging.info("%s %s, gru_input.shape:%s, gru_states:%s" % (filename, self.name, str(gru_input.shape), str(gru_states.shape)))
            return tf.reshape(gru_input, (-1, self.gru_config.gru_units[-1]))
        else:
            raise ValueError("mode:%s not supported." % self.mode)
