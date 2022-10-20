# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/11 6:32 下午
# desc:

import os
import logging
import itertools
from typing import List, Tuple
import tensorflow as tf
from easy_rec_ext.utils import variable_util
from easy_rec_ext.utils.load_class import load_by_path

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
        logging.info("OuterProduct, name:%s, field_num:%d, embed_size:%d, num_pairs:%d" % (self.name, field_num, embed_size, num_pairs))

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
        logging.info("BilinearInteraction, name:%s, field_num:%d, embed_size:%d, bilinear_type:%s" % (self.name, field_num, embed_size, self.bilinear_type))

        if self.bilinear_type == "Field-All":
            W = variable_util.get_normal_variable(
                scope="BilinearInteraction",
                name=self.name + "_weight",
                shape=(embed_size, embed_size)
            )
            vidots = [
                tf.tensordot(
                    tf.slice(input_value, begin=[0, i, 0], size=[-1, 1, -1]),
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
                    tf.slice(input_value, [0, v[1], 0], [-1, 1, -1]),
                )
                for v, w in zip(itertools.combinations(range(field_num), 2), W_list)
            ]
        p = tf.concat(p, axis=1)
        logging.info("BilinearInteraction, name:{name}, p.shape:{shape}".format(name=self.name, shape=str(p.shape)))
        return tf.reshape(p, shape=(-1, field_num * (field_num - 1) // 2 * embed_size))


class CIN(object):
    """
    Compressed Interaction Network used in xDeepFM.
    """

    def __init__(self, name: str,
                 layer_size: Tuple[int] = (128, 128),
                 activation: str = "tf.nn.relu",
                 ):
        self.name = name
        self.layer_size = layer_size
        self.activation = load_by_path(activation)

    def __call__(self, input_value):
        """
        Input shape
            - input_value is a 3D tensor with shape: (batch_size, field_num, embedding_size)
        Output shape
            - 2D tensor with shape: (batch_size, feature_map_num)
        """
        raw_field_num = input_value.get_shape().as_list()[1]
        embed_size = input_value.get_shape().as_list()[2]
        logging.info("CIN, name:%s, raw_field_num:%d, embed_size:%d, layer_size:%s" % (self.name, raw_field_num, embed_size, str(self.layer_size)))

        field_nums = [raw_field_num]
        filters = []
        biases = []
        for i, size in enumerate(self.layer_size):
            filters.append(
                    variable_util.get_normal_variable(
                        scope="CIN",
                        name=self.name + "_weight_%d" % i,
                        shape=[1, field_nums[-1] * field_nums[0], size],
                    )
            )
            biases.append(
                variable_util.get_zero_init_variable(
                    scope="CIN",
                    name=self.name + "_bias_%d" % i,
                    shape=[size],
                )
            )
            field_nums.append(size)

        hidden_nn_layers = [input_value]
        final_result = []

        split_tensor0 = tf.split(hidden_nn_layers[0], embed_size * [1], 2)
        for idx, layer_size in enumerate(self.layer_size):
            split_tensor = tf.split(hidden_nn_layers[-1], embed_size * [1], 2)
            dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True)
            dot_result_o = tf.reshape(dot_result_m, shape=[embed_size, -1, field_nums[0] * field_nums[idx]])
            dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

            curr_out = tf.nn.conv1d(dot_result, filters=filters[idx], stride=1, padding="VALID")
            curr_out = tf.nn.bias_add(curr_out, biases[idx])
            curr_out = self.activation[idx](curr_out)
            curr_out = tf.transpose(curr_out, perm=[0, 2, 1])

            direct_connect = curr_out
            next_hidden = curr_out
            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        result = tf.concat(final_result, axis=1)
        logging.info("CIN, name:{name}, result.shape:{shape}".format(name=self.name, shape=str(result.shape)))
        result = tf.math.reduce_sum(result, -1, keep_dims=False)
        logging.info("CIN, name:{name}, result.shape:{shape}".format(name=self.name, shape=str(result.shape)))
        return result
