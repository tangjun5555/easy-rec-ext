# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/12 2:36 下午
# desc:

import tensorflow as tf
from easy_rec_ext.core.pipeline import TrainConfig


def build(train_config):
    """
    Create distribute training strategy based on config.
    """
    assert isinstance(train_config, TrainConfig)
    # multi-worker strategy with parameter servers
    # under tf1.15 and tf2.x
    if train_config.train_distribute == "TFPSStrategy":
        distribution = tf.distribute.experimental.ParameterServerStrategy()
    else:
        distribution = None
    return distribution
