# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/24 6:52 下午
# desc:

import os
import json
import logging

import tensorflow as tf
from tensorflow.python.keras.layers import GRU

tf = tf.compat.v1

filename = str(os.path.basename(__file__)).split(".")[0]


class SequencePooling(object):
    """
    The SequencePoolingLayer is used to apply pooling operation
    on variable-length sequence feature/multi-value feature.

    Arguments
        name: str.
        mode: str. Pooling operation to be used, can be sum, mean or max.
        gru_config: str.
    """

    def __init__(self, name, mode="mean", gru_config=None):
        self.name = name
        self.mode = mode
        self.gru_config = json.loads(gru_config)

    def __call__(self, seq_value, seq_len):
        """
        Input shape
            - seq_value is a 3D tensor with shape: (batch_size, T, embedding_size)
            - seq_len is a 2D tensor with shape : (batch_size, 1), indicate valid length of each sequence

        Output shape
            - 2D tensor with shape: (batch_size, embedding_size)
        """
        logging.info("%s, seq_value.shape:[%s]" % (filename, ",".join(seq_value.get_shape().as_list())))
        batch_size = seq_value.get_shape().as_list()[0]
        seq_len_max = seq_value.get_shape().as_list()[1]
        embedding_size = seq_value.get_shape().as_list()[2]

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
            gru_units = [int(k) for k in self.gru_config["gru_units"].split(",")]
            go_backwards = self.gru_config["go_backwards"] == 1
            gru_input = seq_value
            for i, j in enumerate(gru_units):
                gru_input, gru_states = GRU(
                    units=j, stateful=True, return_state=True,
                    go_backwards=go_backwards,
                    name='{}_gru_{}'.format(self.name, str(i)),
                )(gru_input)
            return tf.reshape(gru_input, (batch_size, 1, -1))
        else:
            raise ValueError("mode:%s not supported." % self.mode)
