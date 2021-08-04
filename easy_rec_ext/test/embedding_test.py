# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/26 3:40 下午
# desc:

from tensorflow.python.framework import ops

import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1

import tensorflow_recommenders_addons as tfra
from easy_rec_ext.core import embedding_ops

line_sep = "\n" + "##" * 20 + "\n"


def test_06():
    tf.enable_eager_execution()
    weights = embedding_ops.get_embedding_variable(
        "weights", 4
    )
    ids = tf.constant(
        value=[
            [1, 2, 3],
            [4, -1, 0]
        ],
        dtype=tf.dtypes.int64,
    )
    ids = tf.expand_dims(ids, -1)
    values = embedding_ops.safe_embedding_lookup(weights, ids)
    print(line_sep)
    print(ids)
    print(line_sep)
    print(values)


def test_05():
    tf.enable_eager_execution()
    weights = embedding_ops.get_embedding_variable(
        "weights", 4
    )
    ids = tf.constant(
        value=[
            [-1],
            [0],
            [1]
        ],
        dtype=tf.dtypes.int64,
    )

    # sparse_ids = embedding_ops._to_sparse_ids(ids)
    # original_shape = sparse_ids.dense_shape
    # original_rank_dim = sparse_ids.dense_shape.get_shape()[0]
    # print(original_shape)
    # print(original_rank_dim)

    # print(line_sep)
    # print(embedding_ops._to_sparse_ids(ids))

    values = embedding_ops.safe_embedding_lookup(weights, ids)
    print(line_sep)
    print(values)

    # with tf.Session() as sess:
    #     print(sess.run(values))


def test_04():
    tf.disable_eager_execution()

    weights = tfra.dynamic_embedding.get_variable(
        name="w1",
        key_dtype=tf.dtypes.string,
        dim=4,
        initializer=tf.random_normal_initializer(0, 0.1),
        init_size=1,
    )

    ids = tf.SparseTensor(
        indices=[(0, 0), (1, 0), (2, 0), (3, 0)],
        values=tf.constant(value=["431645d82843f859", "431645d82843f859", "", ""], dtype=tf.dtypes.string),
        dense_shape=[5, 1]
    )
    values = tfra.dynamic_embedding.safe_embedding_lookup_sparse(
        embedding_weights=weights,
        sparse_ids=ids
    )

    with tf.Session() as sess:
        print(line_sep)
        print(sess.run(tf.sparse_tensor_to_dense(ids)))
        print(line_sep)
        print(sess.run(values))

        # print(line_sep)
        # print(sess.run(ops.convert_to_tensor(weights)))


def test_03():
    tf.disable_eager_execution()

    w1 = tfra.dynamic_embedding.get_variable(
        name="w1",
        key_dtype=tf.dtypes.int64,
        dim=4,
        initializer=tf.random_normal_initializer(0, 0.1),
    )

    ids2 = tf.SparseTensor(
        indices=[(0, 0), (1, 0), (2, 0), (3, 0)],
        values=tf.constant(value=[1, 1, -1, -1], dtype=tf.dtypes.int64),
        dense_shape=[5, 1]
    )
    v2 = tfra.dynamic_embedding.safe_embedding_lookup_sparse(
        embedding_weights=w1,
        sparse_ids=ids2
    )

    with tf.Session() as sess:
        print(line_sep)
        print(sess.run(tf.sparse_tensor_to_dense(ids2)))
        print(line_sep)
        print(sess.run(v2))

        print(line_sep)
        print(sess.run(ops.convert_to_tensor(w1)))


def test_02():
    tf.disable_eager_execution()

    w1 = tfra.dynamic_embedding.get_variable(
        name="w1",
        key_dtype=tf.dtypes.string,
        dim=4,
        initializer=tf.random_normal_initializer(0, 0.1),
        init_size=1,
    )

    v1 = tfra.dynamic_embedding.embedding_lookup(
        params=w1,
        ids=tf.constant(value=[["431645d82843f859"], ["42fe503ee7820de1ce6420747820813b"]], dtype=tf.dtypes.string)
    )

    v2 = tfra.dynamic_embedding.embedding_lookup(
        params=w1,
        ids=tf.constant(value=[["431645d82843f859"], ["431645d82843f859"]], dtype=tf.dtypes.string)
    )

    v3 = tfra.dynamic_embedding.embedding_lookup(
        params=w1,
        ids=tf.constant(value=[[""], [""]], dtype=tf.dtypes.string)
    )

    with tf.Session() as sess:
        # print(line_sep)
        # print(sess.run(w1))
        print(line_sep)
        print(sess.run(v1))
        print(line_sep)
        print(sess.run(v2))
        print(line_sep)
        print(sess.run(v3))
        # print(line_sep)
        # print(sess.run(tf.convert_to_tensor(w1)))


def test_01():
    tf.enable_eager_execution()

    w1 = tfra.dynamic_embedding.get_variable(
        name="w1",
        dim=4,
        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1),
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
