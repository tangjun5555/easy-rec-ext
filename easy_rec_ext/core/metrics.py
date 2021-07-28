# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import defaultdict

import numpy as np
import tensorflow as tf
from sklearn import metrics as sklearn_metrics

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def _separated_auc_impl(labels, predictions, keys, reduction='mean'):
  """Computes the AUC group by the key separately.

  Args:
    labels: A `Tensor` whose shape matches `predictions`. Will be cast to
      `bool`.
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    keys: keys to be group by, A int or string `Tensor` whose shape matches `predictions`.
    reduction: reduction metric for auc of different keys
      * "mean": simple mean of different keys
      * "mean_by_sample_num": weighted mean with sample num of different keys
      * "mean_by_positive_num": weighted mean with positive sample num of different keys
  """
  assert reduction in ['mean', 'mean_by_sample_num', 'mean_by_positive_num'], \
      'reduction method must in mean | mean_by_sample_num | mean_by_positive_num'
  separated_label = defaultdict(list)
  separated_prediction = defaultdict(list)
  separated_weights = defaultdict(int)

  def update_pyfunc(labels, predictions, keys):
    for label, prediction, key in zip(labels, predictions, keys):
      separated_label[key].append(label)
      separated_prediction[key].append(prediction)
      if reduction == 'mean':
        separated_weights[key] = 1
      elif reduction == 'mean_by_sample_num':
        separated_weights[key] += 1
      elif reduction == 'mean_by_positive_num':
        separated_weights[key] += label

  def value_pyfunc():
    metrics = []
    weights = []
    for key in separated_label.keys():
      per_label = np.asarray(separated_label[key]).reshape([-1])
      per_prediction = np.asarray(separated_prediction[key]).reshape([-1])
      if np.all(per_label == 1) or np.all(per_label == 0):
        continue
      metric = sklearn_metrics.roc_auc_score(per_label, per_prediction)
      metrics.append(metric)
      weights.append(separated_weights[key])
    if len(metrics) > 0:
      return np.average(metrics, weights=weights).astype(np.float32)
    else:
      return np.float32(0.0)

  update_op = tf.py_func(update_pyfunc, [labels, predictions, keys], [])
  value_op = tf.py_func(value_pyfunc, [], tf.float32)
  return value_op, update_op


def gauc(labels, predictions, uids, reduction='mean'):
  """Computes the AUC group by user separately.

  Args:
    labels: A `Tensor` whose shape matches `predictions`. Will be cast to
      `bool`.
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    uids: user ids, A int or string `Tensor` whose shape matches `predictions`.
    reduction: reduction method for auc of different users
      * "mean": simple mean of different users
      * "mean_by_sample_num": weighted mean with sample num of different users
      * "mean_by_positive_num": weighted mean with positive sample num of different users
  """
  return _separated_auc_impl(labels, predictions, uids, reduction)


def pcopc(labels, predictions):
  return tf.reduce_sum(predictions) / tf.reduce_sum(labels)