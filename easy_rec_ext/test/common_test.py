# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/1 12:23 下午
# desc:

import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1

line_sep = "\n" + "##" * 20 + "\n"


def test_01():
    tf.enable_eager_execution()
    t1 = tf.constant(value=[0.0, 1.0, 0.0], dtype=tf.dtypes.float32)
    t2 = tf.to
