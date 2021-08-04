# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/23 5:39 下午
# desc:

import logging
from easy_rec_ext.utils.load_class import load_by_path
import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


class DNN(object):
    def __init__(self, dnn_config, l2_reg, name="dnn", is_training=False):
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
        logging.info("%s activation function = %s" % (self._name, self._config.activation))
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
                name="%s/dnn_%d" % (self._name, i))
            if self._config.use_bn:
                deep_fea = tf.layers.batch_normalization(
                    deep_fea,
                    training=self._is_training,
                    trainable=True,
                    name="%s/dnn_%d/bn" % (self._name, i))
            deep_fea = self.activation(
                deep_fea, name="%s/dnn_%d/act" % (self._name, i))
            if self.dropout_ratio and isinstance(self.dropout_ratio, list) and \
                len(self.dropout_ratio) > 0 and self._is_training:
                assert self.dropout_ratio[
                           i] < 1, "invalid dropout_ratio: %.3f" % self.dropout_ratio[i]
                deep_fea = tf.nn.dropout(
                    deep_fea,
                    keep_prob=1 - self.dropout_ratio[i],
                    name="%s/%d/dropout" % (self._name, i))
        return deep_fea
