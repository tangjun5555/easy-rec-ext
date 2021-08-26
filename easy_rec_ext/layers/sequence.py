# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/24 6:52 下午
# desc:

import logging

import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


class SequencePoolingLayer(object):
    """
    The SequencePoolingLayer is used to apply pooling operation
    on variable-length sequence feature/multi-value feature.

    Arguments
        mode: str. Pooling operation to be used, can be sum, mean or max.
    """

    def __init__(self, name="SequencePoolingLayer", mode="mean"):
        self._name = name
        assert mode in ["sum", "mean", "max"]
        self.mode = mode

    def __call__(self, seq_value, seq_len):
        """
        Input shape
            - seq_value is a 3D tensor with shape: (batch_size, T, embedding_size)
            - seq_len is a 2D tensor with shape : (batch_size, 1), indicate valid length of each sequence

        Output shape
            - 2D tensor with shape: (batch_size, embedding_size)
        """
        seq_len_max = seq_value.shape[1]
        embedding_size = seq_value.shape[-1]

        mask = tf.sequence_mask(seq_len, maxlen=seq_len_max, dtype=tf.dtypes.float32)
        mask = tf.transpose(mask, perm=(0, 2, 1))
        mask = tf.tile(mask, [1, 1, embedding_size])

        if self.mode == "max":
            hist = seq_value - (1 - mask) * 1e9
            return tf.reduce_max(hist, axis=1, keep_dims=True)
        hist = tf.reduce_sum(seq_value * mask, axis=1, keep_dims=False)
        if self.mode == "mean":
            hist = tf.div(hist, tf.cast(seq_len, tf.float32) + 1e-8)
        return hist


class WeightedSequenceLayer(object):
    """
    The WeightedSequenceLayer is used to apply weight score
    on variable-length sequence feature/multi-value feature.

    Arguments
        - weight_normalization: bool. Whether normalize the weight score before applying to sequence.
    """

    def __init__(self, name="WeightedSequenceLayer", weight_normalization=True):
        self._name = name
        self.weight_normalization = weight_normalization

    def __call__(self, seq_value, seq_len, seq_weight):
        """
        Input shape
            - seq_value is a 3D tensor with shape: (batch_size, T, embedding_size)
            - seq_len is a 2D tensor with shape : (batch_size, 1), indicate valid length of each sequence
            - seq_weight is a 2D tensor with shape: (batch_size, T)

        Output shape
            - 2D tensor with shape: (batch_size, embedding_size)

        """
        seq_len_max = seq_value.shape[1]
        embedding_size = seq_value.shape[-1]

        mask = tf.sequence_mask(seq_len, maxlen=seq_len_max, dtype=tf.dtypes.bool)

        if self.weight_normalization:
            paddings = tf.ones_like(seq_weight) * (-2 ** 32 + 1)
        else:
            paddings = tf.zeros_like(seq_weight)
        seq_weight = tf.where(mask, seq_weight, paddings)
        if self.weight_normalization:
            seq_weight = tf.nn.softmax(seq_weight, axis=1)

        seq_weight = tf.expand_dims(seq_weight, axis=2)
        seq_weight = tf.tile(seq_weight, [1, 1, embedding_size])
        return tf.reduce_sum(tf.multiply(seq_value, seq_weight), axis=1, keepdims=False)


class AttentionSequencePoolingLayer(object):
    """
    The Attentional sequence pooling operation.

    Arguments
        - attention_type: str. Attention operation to be used, can be din or scaled_dot.
        - weight_normalization: bool. Whether normalize the attention score.
    """

    def __init__(self,
                 name="AttentionSequencePoolingLayer",
                 attention_type="din",
                 weight_normalization=True,
                 ):
        self._name = name

        self.attention_type = attention_type
        assert self.attention_type in ["din", "scaled_dot"]

        self.weight_normalization = weight_normalization

    def _din(self, query, keys):
        keys_max_len = keys.shape[1]

        cur_ids = tf.tile(query, [1, keys_max_len])
        # (B, seq_max_len, emb_dim)
        cur_ids = tf.reshape(cur_ids, keys.shape)

        # (B, seq_max_len, emb_dim*4)
        net = tf.concat(
            [cur_ids, keys, cur_ids - keys, cur_ids * keys],
            axis=-1,
        )

        name = self._name + "/din"
        with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
            dnn_units = [64, 32, 1]
            for idx, units in enumerate(dnn_units):
                net = tf.layers.dense(
                    net, units=units, activation=tf.nn.relu, name="%s_%d" % (name, idx)
                )

        score = tf.reshape(net, [-1, keys_max_len])
        return score

    def _scaled_dot(self, query, keys):
        keys_max_len = keys.shape[1]

        query = tf.expand_dims(query, axis=1)
        score = tf.matmul(query, keys, transpose_b=True)
        dim = tf.math.sqrt(tf.cast(keys.shape[-1], tf.dtypes.float32))
        score = score / dim

        score = tf.reshape(score, [-1, keys_max_len])
        return score

    def __call__(self, query, keys, keys_length):
        """
        Input shape
            - query is a 2D tensor with shape:  (batch_size, embedding_size)
            - keys is a 3D tensor with shape:   (batch_size, T, embedding_size)
            - keys_length is a 2D tensor with shape: (batch_size, 1)

        Output shape
            - 2D tensor with shape: (batch_size, embedding_size * 2)
        """
        keys_max_len = keys.shape[1]
        # (batch_size, 1, keys_max_len)
        key_masks = tf.sequence_mask(keys_length, maxlen=keys_max_len, dtype=tf.dtypes.bool)

        if self.attention_type == "din":
            attention_score = self._din(query, keys)
        elif self.attention_type == "scaled_dot":
            attention_score = self._scaled_dot(query, keys)
        else:
            raise Exception("attention_type:%s not support." % self.attention_type)
        # (batch_size, keys_max_len) -->  (batch_size, 1, keys_max_len)
        attention_score = tf.reshape(attention_score, [-1, 1, keys_max_len])

        if self.weight_normalization:
            paddings = tf.ones_like(attention_score) * (-2 ** 32 + 1)
        else:
            paddings = tf.zeros_like(attention_score)
        attention_score = tf.where(key_masks, attention_score, paddings)
        if self.weight_normalization:
            attention_score = tf.nn.softmax(attention_score)
        res = tf.matmul(attention_score, keys)
        res = tf.concat([res, query], axis=1)
        return res


class BiLSTM(object):
    """
    A multiple layer Bidirectional Residual LSTM Layer.

    Arguments
        - **units**: Positive integer, dimensionality of the output space.
        - **layers**:Positive integer, number of LSTM layers to stacked.
        - **res_layers**: Positive integer, number of residual connection to used in last ``res_layers``.
        - **dropout_rate**: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.
        - **merge_mode**: Mode by which outputs of the forward and backward RNNs will be combined.
                          One of { ``"fw"`` , ``"bw"`` , ``"sum"`` , ``"mul"`` , ``"concat"`` , ``"ave"`` , ``None`` }.
                          If None, the outputs will not be combined, they will be returned as a list.
    """

    def __init__(self, name="BiLSTM", units=64, layers=2, res_layers=0, dropout_rate=0.2, merge_mode="ave"):
        self._name = name

        self.units = units
        self.layers = layers
        self.res_layers = res_layers
        self.dropout_rate = dropout_rate

        self.merge_mode = merge_mode
        assert merge_mode in ["fw", "bw", "sum", "mul", "ave", "concat", None]

        self.fw_lstm = []
        self.bw_lstm = []
        for _ in range(self.layers):
            self.fw_lstm.append(tf.keras.layers.LSTM(
                self.units, dropout=self.dropout_rate, bias_initializer="ones", return_sequences=True,
                unroll=True,
            ))
            self.bw_lstm.append(tf.keras.layers.LSTM(
                self.units, dropout=self.dropout_rate, bias_initializer="ones", return_sequences=True,
                go_backwards=True, unroll=True,
            ))

    def __call__(self, inputs):
        """
        Input shape
            - 3D tensor with shape (batch_size, timesteps, input_dim)

        Output shape
            - 3D tensor with shape: (batch_size, timesteps, units)

        Returns:

        """
        input_fw = inputs
        input_bw = inputs
        for i in range(self.layers):
            output_fw = self.fw_lstm[i](input_fw)
            output_bw = self.bw_lstm[i](input_bw)
            output_bw = tf.reverse(output_bw, 1)
            if i >= self.layers - self.res_layers:
                output_fw += input_fw
                output_bw += input_bw
            input_fw = output_fw
            input_bw = output_bw

        output_fw = input_fw
        output_bw = input_bw

        if self.merge_mode == "fw":
            output = output_fw
        elif self.merge_mode == "bw":
            output = output_bw
        elif self.merge_mode == "concat":
            output = tf.concat([output_fw, output_bw], axis=-1)
        elif self.merge_mode == "sum":
            output = output_fw + output_bw
        elif self.merge_mode == "ave":
            output = (output_fw + output_bw) / 2
        elif self.merge_mode == "mul":
            output = output_fw * output_bw
        elif self.merge_mode is None:
            output = [output_fw, output_bw]
        else:
            raise Exception("merge_mode:%s not support" % self.merge_mode)
        return output
