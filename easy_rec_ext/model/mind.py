# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/12/23 5:07 PM
# desc:
# MIND: Multi-Interest Network with Dynamic Routing for Recommendation at Tmall

import os
import logging
import tensorflow as tf
from easy_rec_ext.model.dssm import DSSMModel

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


class MINDModelConfig(object):
    pass


class MINDModel(DSSMModel):
    def multi_interest_extractor_layer(self, ):
        pass

    def label_aware_attention_layer(self, multi_query_vectors, item_vector):
        pass
