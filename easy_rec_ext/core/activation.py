# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/26 6:49 下午
# desc:

import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


def parametric_relu(inputs, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable(name="alpha",
                                 shape=inputs.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32,
                                 )
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - tf.math.abs(inputs)) * 0.5
    return pos + neg


def dice(inputs, name, is_training=False, axis=-1, epsilon=1e-9):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        alpha = tf.get_variable(name="alpha",
                                shape=inputs.get_shape()[-1],
                                initializer=tf.constant_initializer(0.0),
                                dtype=tf.float32,
                                )
    inputs_normed = tf.layers.batch_normalization(
        inputs,
        axis=axis,
        epsilon=epsilon,
        name="%s/bn" % name,
        center=False,
        scale=False,
        training=is_training,
        trainable=True,
    )
    x_p = tf.sigmoid(inputs_normed)
    return alpha * (1.0 - x_p) * inputs + x_p * inputs
