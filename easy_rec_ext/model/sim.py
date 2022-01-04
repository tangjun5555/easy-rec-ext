# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/12/23 5:07 PM
# desc: Search-based User Interest Modeling

import os
import logging
import tensorflow as tf
from easy_rec_ext.core import regularizers
from easy_rec_ext.layers import dnn
from easy_rec_ext.model.rank_model import RankModel

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


class SIMModelConfig(object):
    def __init__(self):
        pass

    def __str__(self):
        return str(self.__dict__)


class SIMModelTower(object):
    def __init__(self, input_group: str, sim_config: SIMModelConfig):
        self.input_group = input_group
        self.sim_config = sim_config

    def __str__(self):
        return str(self.__dict__)


class SIMModelLayer(object):
    def call(self, name, deep_fea):
        pass

    def general_search(self, ):
        pass

    def exact_search(self):
        pass
