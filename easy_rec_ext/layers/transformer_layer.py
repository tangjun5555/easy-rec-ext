# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/11/9 11:20 下午
# desc:

import os
import logging
import numpy as np
import tensorflow as tf
from easy_rec_ext.layers.multihead_attention import MultiHeadSelfAttentionConfig, MultiHeadSelfAttention

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


class TransformerEncodeLayer(object):
    def __int__(self, name, multi_head_self_att_config):
        self.name = name
        self.multi_head_self_att_config = multi_head_self_att_config
