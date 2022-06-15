# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/1 6:29 下午
# desc:

from typing import List
import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


def string_to_hash_bucket(input_tensor, num_buckets: int):
    condition = tf.math.logical_or(tf.math.equal(input_tensor, ""), tf.equal(input_tensor, "-1"))
    t1 = tf.string_to_hash_bucket_fast(input_tensor, num_buckets)
    t2 = tf.zeros_like(t1) - 1
    return tf.where(
        condition,
        t2,
        t1
    )


def mapping_by_vocab_list(input_tensor, vocab_list: List[str]):
    assert input_tensor.dtype == tf.dtypes.string
    assert vocab_list and len(vocab_list) == len(set(vocab_list))

    res = tf.zeros_like(input_tensor, dtype=tf.dtypes.int64) - 1
    for i, x in enumerate(vocab_list):
        res = tf.where(
            tf.math.equal(input_tensor, x),
            tf.ones_like(res) * i,
            res
        )
    return res
