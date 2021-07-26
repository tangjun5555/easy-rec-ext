# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/23 5:39 下午
# desc:

import logging
from typing import List
from easy_rec_ext.utils.load_class import load_by_path
import tensorflow as tf

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class DNNConfig(object):
  def __init__(self, hidden_units: List[int], activation: str = "tf.nn.relu", use_bn: bool = False, dropout_ratio=None):
    self.hidden_units = hidden_units
    self.activation = activation
    self.use_bn = use_bn
    self.dropout_ratio = dropout_ratio

  @staticmethod
  def handle(data):
    res = DNNConfig(data["hidden_units"])
    if "activation" in data:
      res.activation = data["activation"]
    if "use_bn" in data:
      res.use_bn = data["use_bn"]
    if "dropout_ratio" in data:
      res.dropout_ratio = data["dropout_ratio"]
    return res

  def __str__(self):
    return str(self.__dict__)


class DNN(object):
  def __init__(self, dnn_config, l2_reg, name='dnn', is_training=False):
    """Initializes a `DNN` Layer.
    Args:
      dnn_config: DNNConfig
      l2_reg: l2 regularizer
      name: scope of the DNN, so that the parameters could be separated from other dnns
      is_training: train phase or not, impact batchnorm and dropout
    """
    self._config = dnn_config
    self._l2_reg = l2_reg
    self._name = name
    self._is_training = is_training
    logging.info('%s activation function = %s' % (self._name, self._config.activation))
    self.activation = load_by_path(self._config.activation)

  @property
  def hidden_units(self):
    return self._config.hidden_units

  @property
  def dropout_ratio(self):
    return self._config.dropout_ratio

  def __call__(self, deep_fea):
    for i, unit in enumerate(self.hidden_units):
      deep_fea = tf.layers.dense(
        inputs=deep_fea,
        units=unit,
        kernel_regularizer=self._l2_reg,
        activation=None,
        name='%s/dnn_%d' % (self._name, i))
      if self._config.use_bn:
        deep_fea = tf.layers.batch_normalization(
          deep_fea,
          training=self._is_training,
          trainable=True,
          name='%s/dnn_%d/bn' % (self._name, i))
      deep_fea = self.activation(
        deep_fea, name='%s/dnn_%d/act' % (self._name, i))
      if len(self.dropout_ratio) > 0 and self._is_training:
        assert self.dropout_ratio[
                 i] < 1, 'invalid dropout_ratio: %.3f' % self.dropout_ratio[i]
        deep_fea = tf.nn.dropout(
          deep_fea,
          keep_prob=1 - self.dropout_ratio[i],
          name='%s/%d/dropout' % (self._name, i))
    return deep_fea
