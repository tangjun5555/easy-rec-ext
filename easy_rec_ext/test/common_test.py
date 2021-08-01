# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/1 12:23 下午
# desc:

import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1

line_sep = "\n" + "##" * 20 + "\n"


def test_05():
    pass


def test_04():
    tf.enable_eager_execution()
    # tf.disable_eager_execution()

    input_tensor = tf.constant(
        value=[
            ["431645d82843f859"],
            ["431645d82843f859"],
            [""],
            [""]
        ],
        dtype=tf.dtypes.string
    )
    num_buckets = 100

    from easy_rec_ext.utils.string_ops import string_to_hash_bucket
    res = string_to_hash_bucket(input_tensor, num_buckets)
    print(line_sep)
    print(res)
    # with tf.Session() as sess:
    #     print(line_sep)
    #     print(sess.run(input_tensor))
    #     print(line_sep)
    #     print(sess.run(res))


def test_03():
    tf.enable_eager_execution()
    t1 = tf.constant(
        value=[
            ["431645d82843f859"],
            ["431645d82843f859"],
            [""],
            [""]
        ],
        dtype=tf.dtypes.string
    )
    t2 = tf.string_to_hash_bucket(t1, 1000)
    t3 = tf.string_to_hash_bucket_fast(t1, 1000)
    t4 = tf.string_to_hash_bucket_strong(t1, 1000, [555, 1234])

    print(line_sep)
    print(t1)
    print(line_sep)
    print(t2)
    print(line_sep)
    print(t3)
    print(line_sep)
    print(t4)

    print(line_sep)
    print(t1 == "")
    # print(line_sep)
    print(tf.ones_like(t1))


def test_02():
    tf.enable_eager_execution()
    t1 = tf.constant(
        value=[[0], [1], [-1]],
        dtype=tf.dtypes.int64,
    )
    t2 = tf.sparse.from_dense(t1)

    print(line_sep)
    print(t1)
    print(line_sep)
    print(t2)


def test_01():
    tf.enable_eager_execution()
    t1 = tf.constant(value=[0.0, 1.0, 0.0], dtype=tf.dtypes.float32)
    t2 = tf.cast(t1, tf.dtypes.bool)

    print(line_sep)
    print(t1)
    print(line_sep)
    print(t2)
