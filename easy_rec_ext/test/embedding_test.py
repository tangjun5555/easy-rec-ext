# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/26 3:40 下午
# desc:

import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1

import tensorflow_recommenders_addons as tfra
from easy_rec_ext.core import embedding_ops

line_sep = "\n" + "##" * 20 + "\n"


def test_06():
    weights = embedding_ops.get_embedding_variable(
        "weights", 4
    )

    # ids = tf.constant(
    #     value=[
    #         [1, 2, 3],
    #         [4, -1, 0]
    #     ],
    #     dtype=tf.dtypes.int64,
    # )
    # ids = tf.expand_dims(ids, -1)
    # values = embedding_ops.safe_embedding_lookup(weights, ids)
    # print(line_sep)
    # print(ids)
    # print(line_sep)
    # print(values)
    #
    # ids = tf.constant(
    #     value=[
    #         [1, ],
    #         [-1],
    #         [0]
    #     ],
    #     dtype=tf.dtypes.int64,
    # )
    # # ids = tf.expand_dims(ids, -1)
    # values = embedding_ops.safe_embedding_lookup(weights, ids)
    # print(line_sep)
    # print(ids)
    # print(line_sep)
    # print(values)

    ids = tf.constant(
        value=[
            [0, 0],
            [1, 0],
            [0, -1]
        ],
        dtype=tf.dtypes.int64,
    )
    ids = tf.expand_dims(ids, -1)
    values = embedding_ops.safe_embedding_lookup(weights, ids)
    print(line_sep)
    print(ids)
    print(line_sep)
    print(values)


def test_02():
    w1 = tfra.dynamic_embedding.get_variable(
        name="w1",
        key_dtype=tf.dtypes.string,
        dim=4,
        initializer=tf.random_normal_initializer(0, 0.1),
        init_size=1,
    )

    # v1 = tfra.dynamic_embedding.embedding_lookup(
    #     params=w1,
    #     ids=tf.constant(value=[["431645d82843f859"], ["42fe503ee7820de1ce6420747820813b"]], dtype=tf.dtypes.string)
    # )
    # print(line_sep)
    # print(v1)
    #
    # v2 = tfra.dynamic_embedding.embedding_lookup(
    #     params=w1,
    #     ids=tf.constant(value=[["431645d82843f859"], ["431645d82843f859"]], dtype=tf.dtypes.string)
    # )
    # print(line_sep)
    # print(v2)

    # v3 = tfra.dynamic_embedding.embedding_lookup(
    #     params=w1,
    #     ids=tf.constant(value=[[""], [""]], dtype=tf.dtypes.string)
    # )
    v3 = embedding_ops.safe_embedding_lookup(
        params=w1,
        ids=tf.constant(value=[[""], ["431645d82843f859"], ["431645d82843f859"], ["42fe503ee7820de1ce6420747820813b"]], dtype=tf.dtypes.string)
    )
    print(line_sep)
    print(v3)

    # v4 = tfra.dynamic_embedding.embedding_lookup_unique(
    #     params=w1,
    #     ids=tf.constant(value=["431645d82843f859", "431645d82843f859"], dtype=tf.dtypes.string)
    # )
    # print(line_sep)
    # print(v4)


def test_01():
    w1 = tfra.dynamic_embedding.get_variable(
        name="w1",
        dim=4,
        initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0),
    )
    print(line_sep)
    print(w1)

    v1 = tfra.dynamic_embedding.embedding_lookup_unique(
        params=w1,
        ids=tf.constant(value=[[1], [100]], dtype=tf.dtypes.int64)
    )
    print(line_sep)
    print(v1)

    v2 = tfra.dynamic_embedding.embedding_lookup_unique(
        params=w1,
        ids=tf.constant(value=[[1], [1]], dtype=tf.dtypes.int64)
    )
    print(line_sep)
    print(v2)

    v3 = tfra.dynamic_embedding.embedding_lookup_unique(
        params=w1,
        ids=tf.constant(value=[[-1], [-1]], dtype=tf.dtypes.int64)
    )
    print(line_sep)
    print(v3)

    v4 = tfra.dynamic_embedding.embedding_lookup_unique(
        params=w1,
        ids=tf.constant(value=[[1, 1], [2, 2]], dtype=tf.dtypes.int64)
    )
    print(line_sep)
    print(v4)
