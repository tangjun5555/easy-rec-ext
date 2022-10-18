# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2022/10/18 17:56
# desc:

import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


def numerical_conversion_fn(fn_type, input_values):
    if fn_type == "square":
        res = tf.math.square(input_values)
        return tf.math.multiply(tf.math.sign(input_values), res)
    elif fn_type == "sqrt":
        res = tf.math.sqrt(
            tf.math.multiply(tf.math.sign(input_values), input_values)
        )
        return tf.math.multiply(tf.math.sign(input_values), res)
    elif fn_type == "log":
        res = tf.math.log(
            tf.math.multiply(tf.math.sign(input_values), input_values) + 1.0
        )
        return tf.math.multiply(tf.math.sign(input_values), res)
    elif fn_type == "half":
        return tf.math.divide(input_values, 2.0)
    else:
        raise ValueError("fn:{fn_type} not supported.".format(fn_type=fn_type))
