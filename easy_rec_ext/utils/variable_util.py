# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/2 2:08 下午
# desc:

import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


def get_normal_variable(scope, name, shape):
    """
    获取矩阵变量
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        variable = tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1),
            dtype=tf.float32,
        )
    return variable
