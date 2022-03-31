# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/24 6:52 下午
# desc:

import os
import logging
from typing import List
import tensorflow as tf
from easy_rec_ext.layers.multihead_attention import MultiHeadSelfAttention

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


class RNNConfig(object):
    def __init__(self, hidden_units: List[int], go_backwards: bool = False):
        assert hidden_units and hidden_units[0] > 0
        self.hidden_units = hidden_units
        self.go_backwards = go_backwards

    @staticmethod
    def handle(data):
        res = RNNConfig(data["hidden_units"])
        if "go_backwards" in data:
            res.go_backwards = data["go_backwards"]
        return res


class MultiHeadSelfAttentionConfig(object):
    def __init__(self, head_num, head_size, use_res=False):
        self.head_num = head_num
        self.head_size = head_size
        self.use_res = use_res

    @staticmethod
    def handle(data):
        res = MultiHeadSelfAttentionConfig(data["head_num"], data["head_size"])
        if "use_res" in data:
            res.use_res = data["use_res"]
        return res


class SequencePoolingConfig(object):
    def __init__(self, mode: str = "sum",
                 gru_config: RNNConfig = None,
                 lstm_config: RNNConfig = None,
                 self_att_config: MultiHeadSelfAttentionConfig = None,
                 ):
        self.mode = mode
        self.gru_config = gru_config
        self.lstm_config = lstm_config
        self.self_att_config = self_att_config

    @staticmethod
    def handle(data):
        res = SequencePoolingConfig()
        if "mode" in data:
            res.mode = data["mode"]
        if "gru_config" in data:
            res.gru_config = RNNConfig.handle(data["gru_config"])
        if "lstm_config" in data:
            res.lstm_config = RNNConfig.handle(data["lstm_config"])
        if "self_att_config" in data:
            res.self_att_config = MultiHeadSelfAttentionConfig.handle(data["self_att_config"])
        return res


class SequencePooling(object):
    """
    The SequencePoolingLayer is used to apply pooling operation
    on variable-length sequence feature/multi-value feature.

    Arguments
        name: str.
        mode: str. Pooling operation to be used.
    """

    def __init__(self, name, mode="sum",
                 gru_config: RNNConfig = None,
                 lstm_config: RNNConfig = None,
                 self_att_config: MultiHeadSelfAttentionConfig = None
                 ):
        self.name = name
        self.mode = mode
        self.gru_config = gru_config
        self.lstm_config = lstm_config
        self.self_att_config = self_att_config

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
            if len(self.gru_config.hidden_units) > 1:
                for i, j in enumerate(self.gru_config.hidden_units[:-1]):
                    gru_input = tf.keras.layers.GRU(
                        units=j,
                        return_sequences=True,
                        go_backwards=self.gru_config.go_backwards,
                        name="{}_gru_{}".format(self.name, str(i)),
                    )(gru_input)
                    logging.info(
                        "%s %s, i:%d, j:%d, gru_input.shape:%s" % (filename, self.name, i, j, str(gru_input.shape)))
            gru_input = tf.keras.layers.GRU(
                units=self.gru_config.hidden_units[-1],
                go_backwards=self.gru_config.go_backwards,
                name="{}_gru_{}".format(self.name, str(len(self.gru_config.hidden_units) - 1)),
            )(gru_input)
            logging.info("%s %s, gru_input.shape:%s" % (filename, self.name, str(gru_input.shape)))
            return gru_input
        elif self.mode == "lstm":
            lstm_input = seq_value
            if len(self.lstm_config.hidden_units) > 1:
                for i, j in enumerate(self.lstm_config.hidden_units[:-1]):
                    lstm_input = tf.keras.layers.LSTM(
                        units=j,
                        return_sequences=True,
                        go_backwards=self.lstm_config.go_backwards,
                        name="{}_lstm_{}".format(self.name, str(i)),
                    )(lstm_input)
                    logging.info(
                        "%s %s, i:%d, j:%d, lstm_input.shape:%s" % (filename, self.name, i, j, str(lstm_input.shape)))
            lstm_input = tf.keras.layers.LSTM(
                units=self.lstm_config.hidden_units[-1],
                go_backwards=self.lstm_config.go_backwards,
                name="{}_lstm_{}".format(self.name, str(len(self.lstm_config.hidden_units) - 1)),
            )(lstm_input)
            logging.info("%s %s, lstm_input.shape:%s" % (filename, self.name, str(lstm_input.shape)))
            return lstm_input
        elif self.mode == "self_att":
            self_att_output = MultiHeadSelfAttention(
                name=self.name + "_" + "self_att",
                head_num=self.self_att_config.head_num,
                head_size=self.self_att_config.head_size,
                l2_reg=None,
                use_res=self.self_att_config.use_res,
            )(seq_value)
            logging.info("%s %s, self_att_output.shape:%s" % (filename, self.name, str(self_att_output.shape)))
            self_att_output = tf.reduce_sum(self_att_output, axis=1, keepdims=False)
            return self_att_output
        else:
            raise ValueError("mode:%s not supported." % self.mode)
