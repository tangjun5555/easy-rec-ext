# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/11 6:32 下午
# desc:

import os
import logging

import tensorflow as tf
from easy_rec_ext.core import embedding_ops

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
        Input shape
            - seq_value is a 3D tensor with shape: (batch_size, field_num, embedding_size)

        Output shape
            - 2D tensor with shape: (batch_size, pairs)
        """
        field_num = input_value.get_shape().as_list()[1]
        embed_size = input_value.get_shape().as_list()[2]
        logging.info("InnerProduct, name:%s, field_num:%d, embed_size:%d" % (self.name, field_num, embed_size))

        row = []
        col = []
        for i in range(field_num - 1):
            for j in range(i + 1, field_num):
                row.append(i)
                col.append(j)

        p = tf.concat(
            [tf.slice(input_value, [0, idx, 0], [-1, 1, -1]) for idx in row],
            axis=1,
        )
        q = tf.concat(
            [tf.slice(input_value, [0, idx, 0], [-1, 1, -1]) for idx in col],
            axis=1,
        )
        logging.info("InnerProduct, name:%s, p.shape:%s, q.shape:%s" % (self.name, str(p.shape), str(q.shape)))

        inner_product = tf.multiply(p, q)
        inner_product = tf.reduce_sum(inner_product, axis=2, keep_dims=False)
        return inner_product


class CAN(object):
    def __init__(self, name, item_vocab_size, order=2, mlp_units=(8, 4)):
        """
        Co-action Network
        Args:
            name:
        """
        self.name = name
        self.item_vocab_size = item_vocab_size
        self.order = order
        self.mlp_units = mlp_units

    def __call__(self, user_value, item_value):
        """
        Args:
            user_value:
                3D tensor with shape: (batch_size, user_field_num, embedding_size)
                fed into MLP
            item_value:
                2D tensor with shape: (batch_size, 1), use the original id
                server as the weight and bias
        Returns:
            - 2D tensor with shape: (batch_size, order * mlp_units[-1])
        """
        user_dimension = user_value.get_shape().as_list()[-1]
        item_dimension = 0
        weight_emb_w = []
        weight_emb_b = []
        x = user_dimension
        for y in self.mlp_units:
            item_dimension += x * y
            item_dimension += 1

            weight_emb_w.append([x, y])
            weight_emb_b.append(1)

            x = y

        item_embedding_weight = embedding_ops.get_embedding_variable(
            name=self.name + "_item",
            dim=item_dimension,
            vocab_size=self.item_vocab_size
        )
        item_embedding_value = embedding_ops.safe_embedding_lookup(
            item_embedding_weight, item_value,
        )

        weight, bias = [], []
        idx = 0
        weight_orders = []
        bias_orders = []
        for i in range(self.order):
            for w, b in zip(weight_emb_w, weight_emb_b):
                weight.append(tf.slice(item_embedding_value, [0, idx], [-1, w[0] * w[1]]))
                idx += w[0] * w[1]
                bias.append(tf.slice(item_embedding_value, [0, idx], [-1, b]))
                idx += b
            weight_orders.append(weight)
            bias_orders.append(bias)

        out_seq = []
        hh = [user_value]
        for i in range(self.order - 1):
            hh.append(tf.multiply(hh[-1] ** user_value))
        for i, h in enumerate(hh):
            weight, bias = weight_orders[i], bias_orders[i]
            for j, (w, b) in enumerate(zip(weight, bias)):
                h = tf.matmul(h, w)
                h = h + b
                if j != len(weight) - 1:
                    h = tf.nn.tanh(h)
            out_seq.append(h)
        out_seq = tf.concat(out_seq, 2)

        # if mask is not None:
        #     mask = tf.expand_dims(mask, axis=-1)
        #     out_seq = out_seq * mask
        return tf.reduce_sum(out_seq, 1, keepdims=False)
