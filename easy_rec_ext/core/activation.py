# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/26 6:49 下午
# desc:

import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


def parametric_relu(_x, name=None):
    assert name
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable("alpha", _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32,
                                 )
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg


def dice(_x, axis=-1, epsilon=1e-9, name=None):
    assert name
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable("alpha", _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32
                                 )
        input_shape = list(_x.get_shape())
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[axis] = input_shape[axis]
    mean = tf.reduce_mean(_x, axis=reduction_axes)
    brodcast_mean = tf.reshape(mean, broadcast_shape)
    std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axes)
    std = tf.sqrt(std)
    brodcast_std = tf.reshape(std, broadcast_shape)
    x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
    # x_normed = tf.layers.batch_normalization(_x, center=False, scale=False)
    x_p = tf.sigmoid(x_normed)
    return alphas * (1.0 - x_p) * _x + x_p * _x
