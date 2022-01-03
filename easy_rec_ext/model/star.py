# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2022/1/3 10:13 AM
# desc: Star Topology Adaptive Recommender


import os
import logging
from typing import List, Dict
from collections import OrderedDict
import tensorflow as tf
from easy_rec_ext.layers import dnn
from easy_rec_ext.model.multi_tower import MultiTower
import easy_rec_ext.core.metrics as metrics_lib

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


class STARModelConfig(object):
    pass
