# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/30 4:57 下午
# desc:

import logging
from easy_rec_ext.input.input import Input

import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


class TFRecordInput(Input):
    def __init__(self, input_config, feature_config, input_path):
        pass

    def _build(self, mode):
        pass
