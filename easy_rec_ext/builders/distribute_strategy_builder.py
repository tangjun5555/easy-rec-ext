# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/12 2:36 下午
# desc:

import tensorflow as tf
from easy_rec_ext.core.pipeline import TrainConfig


def build(train_config):
    """
    Create distribute training strategy based on config.
    Args:
        train_config:

    Returns:

    """
    assert isinstance(train_config, TrainConfig)
    # single worker multi-gpu strategy
    if train_config.train_distribute == "MirroredStrategy":
        distribution = tf.distribute.MirroredStrategy()
    # multi worker multi-gpu strategy
    # works under tf1.15 and tf2.x
    elif train_config.train_distribute == "MultiWorkerMirroredStrategy":
        distribution = tf.distribute.MultiWorkerMirroredStrategy()
    # works under tf1.15 and tf2.x
    elif train_config.train_distribute == "PSStrategy":
        distribution = tf.distribute.experimental.ParameterServerStrategy()
    else:
        raise ValueError("train_distribute:%s not supported." % train_config.train_distribute)
    return distribution
