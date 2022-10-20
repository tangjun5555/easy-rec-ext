# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/2 2:08 下午
# desc:

import numpy as np
import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


def load_variable_by_file(filename):
    values = []
    with open(filename, mode="r") as f:
        for line in f:
            tmp = [float(x) for x in line.strip().split(",")]
            values.append(tmp)
    return np.array(values)


def get_normal_variable(scope, name, shape):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        variable = tf.get_variable(
            name=name,
            shape=shape,
            dtype=tf.dtypes.float32,
            initializer=tf.glorot_uniform_initializer(seed=555),
            trainable=True,
        )
    return variable


def get_zero_init_variable(scope, name, shape):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        variable = tf.get_variable(
            name=name,
            shape=shape,
            dtype=tf.dtypes.float32,
            initializer=tf.zeros_initializer(),
            trainable=True,
        )
    return variable
