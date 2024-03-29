# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/4 2:31 下午
# desc:

import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


class LayerNormalization(object):
    """
    Layer normalization for BTC format: supports L2(default) and L1 modes.
    """

    def __init__(self, hidden_size, name, norm_type="layernorm_L2"):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size
        self.name = name
        self.norm_type = norm_type
        self.epsilon = 1e-6

        self.scale = tf.get_variable(
            self.name + "/" + "layer_norm_scale",
            [self.hidden_size],
            initializer=tf.keras.initializers.Ones(),
            dtype=tf.float32
        )
        self.bias = tf.get_variable(
            self.name + "/" + "layer_norm_bias",
            [self.hidden_size],
            initializer=tf.keras.initializers.Zeros(),
            dtype=tf.float32
        )

    def __call__(self, x):
        if self.norm_type == "layernorm_L2":
            dtype = x.dtype
            x = tf.cast(x=x, dtype=tf.float32)
            mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
            variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
            norm_x = (x - mean) * tf.rsqrt(variance + self.epsilon)
            result = norm_x * self.scale + self.bias
            return tf.cast(x=result, dtype=dtype)
        else:
            dtype = x.dtype
            if dtype == tf.float16:
                x = tf.cast(x, dtype=tf.float32)
            mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
            x = x - mean
            variance = tf.reduce_mean(tf.abs(x), axis=[-1], keepdims=True)
            norm_x = tf.div(x, variance + self.epsilon)
            y = norm_x * self.scale + self.bias
            if dtype == tf.float16:
                y = tf.saturate_cast(y, dtype)
            return y
