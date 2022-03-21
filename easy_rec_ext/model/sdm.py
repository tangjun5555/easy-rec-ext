# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/19 6:35 下午
# desc:
# Reference:
# SDM: Sequential Deep Matching Model for Online Large-scale Recommender System

import os
import logging
import tensorflow as tf
from easy_rec_ext.layers import dnn
from easy_rec_ext.model.match_model import MatchModel

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


class SDMModelConfig(object):
    def __init__(self):
        pass


class SDMModel(object):
    pass


class SDM(MatchModel, SDMModel):
    def __init__(self,
                 model_config,
                 feature_config,
                 features,
                 labels=None,
                 is_training=False,
                 ):
        super(SDM, self).__init__(model_config, feature_config, features, labels, is_training)
