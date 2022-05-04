# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2022/5/4 3:38 PM
# desc:

import os
import logging
import tensorflow as tf
from easy_rec_ext.core import embedding_ops
from easy_rec_ext.layers.common_layers import leaky_relu

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


class AutoDis(object):
    def __init__(self, name, in_dim, each_out_size, meta_emb_num=5):
        self.name = name
        self.in_dim = in_dim
        self.each_out_size = each_out_size
        self.meta_emb_num = meta_emb_num

    def __call__(self, input_value):
        """
        Input shape
            - input_value is a 3D tensor with shape: (batch_size, in_dim)
        Output shape
            - 2D tensor with shape: (batch_size, in_dim * each_out_size)
        """
        # Meta-Embeddings
        meta_embedding = embedding_ops.get_embedding_variable(
            name=self.name + "_meta_embeddings",
            dim=self.each_out_size,
            vocab_size=self.in_dim * self.meta_emb_num,
            key_is_string=False,
        )
        meta_embedding = tf.reshape(meta_embedding, [self.in_dim, self.meta_emb_num, self.each_out_size])
        meta_embedding = tf.expand_dims(meta_embedding, axis=0)
        meta_embedding = tf.tile(meta_embedding, [tf.shape(input_value)[0], 1, 1, 1])
        logging.info("%s %s, meta_embedding.shape:%s" % (filename, self.name, str(meta_embedding.shape)))

        # Automatic Discretization
        h = tf.layers.dense(
            tf.expand_dims(input_value, axis=-1),
            self.each_out_size,
            use_bias=False,
            name=self.name + "_mlp1"
        )
        h = leaky_relu(h)
        score = tf.layers.dense(
            h,
            self.each_out_size,
            use_bias=False,
            name=self.name + "_mlp2"
        )
        score = score + h  # skip-connection
        score = tf.nn.softmax(score)
        score = tf.expand_dims(score, axis=-1)
        logging.info("%s %s, score.shape:%s" % (filename, self.name, str(score.shape)))

        # Aggregation Function
        output = score * meta_embedding
        output = tf.reduce_sum(output, axis=-1, keepdims=False)
        output = tf.reshape(output, [-1, self.in_dim * self.each_out_size])
        return output
