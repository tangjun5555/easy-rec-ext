# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/1 6:29 下午
# desc:

import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


def string_to_hash_bucket(input_tensor, num_buckets):
    condition = tf.math.logical_or(tf.equal(input_tensor, ""), tf.equal(input_tensor, "-1"))
    t1 = tf.string_to_hash_bucket_fast(input_tensor, num_buckets)
    t2 = tf.zeros_like(t1) - 1
    return tf.where(
        condition,
        t2,
        t1
    )
