# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2022/4/3 10:10 PM
# desc:

import os
import logging
import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


class ConcatAttentionScore(object):
    def __init__(self, name, scale=True):
        self.name = name
        self.scale = scale

    def __call__(self, query, key):
        """
        Args:
            query: [batch_size, embed_size]
            key: [batch_size, T, embed_size]
        Returns:
            score: [batch_size, 1, T]
        """
        key_shape = key.get_shape().as_list()
        query = tf.tile(query, [1, key_shape[0]])
        query = tf.reshape(query, key_shape)

        q_k = tf.concat([query, key], axis=-1)
        output = tf.layers.dense(
            q_k, 1,
            activation=tf.nn.tanh,
            name=self.name + "_" + "dense"
        )
        if self.scale:
            output = output / (key.get_shape().as_list()[-1] ** 0.5)
        return output


class ProductAttentionScore(object):
    def __init__(self, name, scale=True):
        self.name = name
        self.scale = scale

    def __call__(self, query, key):
        """
        Args:
            query: [batch_size, embed_size]
            key: [batch_size, T, embed_size]

        Returns:
            score: [batch_size, 1, T]
        """
        output = tf.matmul(tf.expand_dims(query, axis=1), tf.transpose(key, [0, 2, 1]))
        if self.scale:
            output = output / (key.get_shape().as_list()[-1] ** 0.5)
        return output


class TargetAttention(object):
    def __init__(self, name, score_type="product"):
        self.name = name
        self.score_type = score_type

    def __call__(self, query, key, key_length):
        """
        Args:
            query: [batch_size, embed_size]
            key: [batch_size, T, embed_size]
            key_length: [batch_size]
        Returns:
            output: [batch_size, embed_size]
        """
        if self.score_type == "concat":
            att_score = ConcatAttentionScore(
                name=self.name + "_" + "score"
            )(query, key)
        else:
            att_score = ProductAttentionScore(
                name=self.name + "_" + "score"
            )(query, key)

        max_len = tf.shape(key)[1]
        padding = tf.ones_like(att_score) * (-2 ** 32 + 1)
        key_masks = tf.sequence_mask(tf.expand_dims(key_length, axis=-1), max_len)
        att_score = tf.where(key_masks, att_score, padding)
        att_score = tf.nn.softmax(att_score)

        output = tf.matmul(att_score, key)
        return tf.squeeze(output, axis=1)
