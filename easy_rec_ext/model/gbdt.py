# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2022/2/21 11:56 AM
# desc:

import os
import logging
import tensorflow as tf
from tensorflow_estimator.python.estimator.canned import boosted_trees

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


class GBDTLayer(object):
    def call(self, name, deep_fea):
        pass
