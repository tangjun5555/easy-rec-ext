# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/11 6:32 下午
# desc:

import os
import logging
import itertools
import tensorflow as tf
from easy_rec_ext.utils import variable_util

if tf.__version__ >= "2.0":
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
            - input_value is a 3D tensor with shape: (batch_size, field_num, embedding_size)

        Output shape
            - 2D tensor with shape: (batch_size, 1)
        """
        field_num = input_value.get_shape().as_list()[1]
        embed_size = input_value.get_shape().as_list()[2]
        logging.info("FM, name:%s, field_num:%d, embed_size:%d" % (self.name, field_num, embed_size))

        sum_square = tf.square(tf.reduce_sum(input_value, 1, keepdims=False))
        square_sum = tf.reduce_sum(tf.square(input_value), 1, keepdims=False)
        return 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum))


class FFM(object):
    def __init__(self, name):
        self.name = name

    def __call__(self, input_value):
        """
        Input shape
            - input_value is a 3D tensor with shape: (batch_size, field_num, embedding_size)

        Output shape
            - 2D tensor with shape: (batch_size, 1)
        """
        field_num = input_value.get_shape().as_list()[1]
        embed_size = input_value.get_shape().as_list()[2]
        logging.info("FFM, name:%s, field_num:%d, embed_size:%d" % (self.name, field_num, embed_size))


class InnerProduct(object):
    def __init__(self, name):
        self.name = name

    def __call__(self, input_value):
        """
        Input shape
            - input_value is a 3D tensor with shape: (batch_size, field_num, embedding_size)

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


class OuterProduct(object):
    def __init__(self, name):
        self.name = name

    def __call__(self, input_value):
        """
        Input shape
            - input_value is a 3D tensor with shape: (batch_size, field_num, embedding_size)

        Output shape
            - 2D tensor with shape: (batch_size, pairs)
        """
        field_num = input_value.get_shape().as_list()[1]
        embed_size = input_value.get_shape().as_list()[2]
        num_pairs = int(field_num * (field_num - 1) / 2)
        logging.info("OuterProduct, name:%s, field_num:%d, embed_size:%d, num_pairs:%d" % (
        self.name, field_num, embed_size, num_pairs))

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
        logging.info("OuterProduct, name:%s, p.shape:%s, q.shape:%s" % (self.name, str(p.shape), str(q.shape)))

        kernel = variable_util.get_normal_variable(
            scope="OuterProduct_" + self.name,
            name="kernel",
            shape=(num_pairs, embed_size)
        )
        kernel = tf.expand_dims(kernel, 0)

        kp = tf.math.reduce_sum(p * q * kernel, -1)
        return kp


class BilinearInteraction(object):
    """
    BilinearInteraction Layer used in FiBiNET
    """
    def __init__(self, name, bilinear_type="Field-Interaction"):
        self.name = name
        assert bilinear_type in ["Field-All", "Field-Each", "Field-Interaction"]
        self.bilinear_type = bilinear_type

    def __call__(self, input_value):
        """
        Input shape
            - input_value is a 3D tensor with shape: (batch_size, field_num, embedding_size)
        Output shape
            - 2D tensor with shape: (batch_size, pairs * embedding_size)
        """
        field_num = input_value.get_shape().as_list()[1]
        embed_size = input_value.get_shape().as_list()[2]

        if self.bilinear_type == "Field-All":
            W = variable_util.get_normal_variable(
                scope="BilinearInteraction",
                name=self.name + "_weight",
                shape=(embed_size, embed_size)
            )
            vidots = [
                tf.tensordot(
                    tf.slice(input_value, [0, i, 0], [-1, 1, -1]),
                    W,
                    axes=(-1, 0),
                )
                for i in range(field_num - 1)
            ]
            p = [
                tf.multiply(
                    vidots[i],
                    tf.slice(input_value, [0, j, 0], [-1, 1, -1]),
                )
                for i, j in itertools.combinations(range(field_num), 2)
            ]
        elif self.bilinear_type == "Field-Each":
            W_list = [
                variable_util.get_normal_variable(
                    scope="BilinearInteraction",
                    name=self.name + "_weight_%d" % i,
                    shape=(embed_size, embed_size)
                )
                for i in range(field_num - 1)
            ]
            vidots = [
                tf.tensordot(
                    tf.slice(input_value, [0, i, 0], [-1, 1, -1]),
                    W_list[i],
                    axes=(-1, 0),
                )
                for i in range(field_num - 1)
            ]
            p = [
                tf.multiply(
                    vidots[i],
                    tf.slice(input_value, [0, j, 0], [-1, 1, -1]),
                )
                for i, j in itertools.combinations(range(field_num), 2)
            ]
        else:
            W_list = [
                variable_util.get_normal_variable(
                    scope="BilinearInteraction",
                    name=self.name + "_weight_%d_%d" % (i, j),
                    shape=(embed_size, embed_size)
                )
                for i, j in itertools.combinations(range(field_num), 2)
            ]
            p = [
                tf.multiply(
                    tf.tensordot(tf.slice(input_value, [0, v[0], 0], [-1, 1, -1]), w, axes=(-1, 0)),
                    tf.slice(input_value, [0, v[1], 0], [-1, 1, -1])
                )
                for v, w in zip(itertools.combinations(range(field_num), 2), W_list)
            ]
        return tf.reshape(p, shape=(-1, field_num * (field_num - 1) / 2 * embed_size))
