# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/26 3:40 下午
# desc:

import tensorflow as tf

if tf.__version__ >= '2.0':
    tf = tf.compat.v1

import tensorflow_recommenders_addons as tfra

line_sep = "\n" + "##" * 20 + "\n"


def test_02():
    tf.disable_eager_execution()

    w1 = tfra.dynamic_embedding.get_variable(
        name="w1",
        key_dtype=tf.dtypes.string,
        dim=4,
        initializer=tf.random_normal_initializer(0, 0.1),
        init_size=16,
    )
    # print(line_sep)
    # print(w1)

    v1 = tfra.dynamic_embedding.embedding_lookup(
        params=w1,
        ids=tf.constant(value=[["431645d82843f859"], ["42fe503ee7820de1ce6420747820813b"]], dtype=tf.dtypes.string)
    )
    # print(line_sep)
    # print(v1)

    v2 = tfra.dynamic_embedding.embedding_lookup(
        params=w1,
        ids=tf.constant(value=[["431645d82843f859"], ["431645d82843f859"]], dtype=tf.dtypes.string)
    )
    # print(line_sep)
    # print(v2)

    v3 = tfra.dynamic_embedding.embedding_lookup(
        params=w1,
        ids=tf.constant(value=[[""], [""]], dtype=tf.dtypes.string)
    )
    # print(line_sep)
    # print(v3)

    with tf.Session() as sess:
        print(line_sep)
        print(sess.run(w1))
        print(line_sep)
        print(sess.run(v1))
        print(line_sep)
        print(sess.run(v2))
        print(line_sep)
        print(sess.run(v3))


def test_01():
    tf.enable_eager_execution()

    w1 = tfra.dynamic_embedding.get_variable(
        name="w1",
        dim=4,
        initializer=tf.random_normal_initializer(0, 0.1),
    )
    print(line_sep)
    print(w1)

    v1 = tfra.dynamic_embedding.embedding_lookup(
        params=w1,
        ids=tf.constant(value=[[1], [100]], dtype=tf.dtypes.int64)
    )
    print(line_sep)
    print(v1)

    v2 = tfra.dynamic_embedding.embedding_lookup(
        params=w1,
        ids=tf.constant(value=[[1], [1]], dtype=tf.dtypes.int64)
    )
    print(line_sep)
    print(v2)

    v3 = tfra.dynamic_embedding.embedding_lookup(
        params=w1,
        ids=tf.constant(value=[[-1], [-1]], dtype=tf.dtypes.int64)
    )
    print(line_sep)
    print(v3)

    v4 = tfra.dynamic_embedding.embedding_lookup(
        params=w1,
        ids=tf.constant(value=[[1, 1], [2, 2]], dtype=tf.dtypes.int64)
    )
    print(line_sep)
    print(v4)
