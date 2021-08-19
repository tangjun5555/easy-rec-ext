# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/11 6:32 下午
# desc:

import logging

import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


class FM(object):
    def __init__(self, name="fm", field_num=None, embed_size=None):
        """
        Initializes a `FM` Layer.
        Args:
          name: scope of the FM
        """
        self._name = name
        self._field_num = field_num
        self._embed_size = embed_size

        assert self._field_num or self._embed_size

    def __call__(self, fm_fea):
        shape = fm_fea.get_shape().as_list()
        assert len(shape) == 2
        field_num = self._field_num if self._field_num else shape[1] // self._embed_size
        embed_size = shape[1] // field_num
        assert field_num * embed_size == shape[1]
        logging.info("fm, name:%s, field_num:%d, embed_size:%d" % (self._name, field_num, embed_size))
        with tf.name_scope(self._name):
            fm_feas = tf.reshape(fm_fea, [-1, field_num, embed_size])
            sum_square = tf.square(tf.reduce_sum(fm_feas, 1))
            square_sum = tf.reduce_sum(tf.square(fm_feas), 1)
            y_v = 0.5 * tf.subtract(sum_square, square_sum)
        return y_v
