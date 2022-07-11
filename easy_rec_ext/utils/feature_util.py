# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2022/7/11 3:42 PM
# desc:

import tensorflow as tf
from easy_rec_ext.utils import string_ops

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


def compute_seq_fea_len(hist_seq_id):
    # TODO Maybe more elegant
    if hist_seq_id.dtype == tf.dtypes.string:
        hist_seq_len = tf.where(string_ops.compute_invalid_string_id_condition(hist_seq_id),
                                tf.zeros(tf.shape(hist_seq_id), dtype=tf.dtypes.int32),
                                tf.ones(tf.shape(hist_seq_id), dtype=tf.dtypes.int32)
                                )
    else:
        hist_seq_len = tf.where(tf.less(hist_seq_id, 0), tf.zeros_like(hist_seq_id), tf.ones_like(hist_seq_id))
    return tf.reduce_sum(hist_seq_len, axis=1, keep_dims=False)
