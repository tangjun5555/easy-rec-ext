# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/12/23 5:07 PM
# desc:
# MIND: Multi-Interest Network with Dynamic Routing for Recommendation at Tmall

import os
import logging
import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


class MINDModelConfig(object):
    pass


