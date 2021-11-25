# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/27 12:19 下午
# desc:

from collections import defaultdict

import numpy as np
import tensorflow as tf
from sklearn import metrics as sklearn_metrics

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


def gauc(labels, predictions, gids, reduction="mean_by_sample_num"):
    """Computes the AUC group by user separately.
  
    Args:
      labels: A `Tensor` whose shape matches `predictions`. Will be cast to
        `bool`.
      predictions: A floating point `Tensor` of arbitrary shape and whose values
        are in the range `[0, 1]`.
      gids: group ids, A int or string `Tensor` whose shape matches `predictions`.
      reduction: reduction method for auc of different users
        * "mean": simple mean of different users
        * "mean_by_sample_num": weighted mean with sample num of different users
        * "mean_by_positive_num": weighted mean with positive sample num of different users
    """
    assert reduction in ["mean", "mean_by_sample_num", "mean_by_positive_num"], \
        "reduction method must in mean | mean_by_sample_num | mean_by_positive_num"

    separated_label = defaultdict(list)
    separated_prediction = defaultdict(list)
    separated_weights = defaultdict(int)

    def update_pyfunc(labels, predictions, keys):
        for label, prediction, key in zip(labels, predictions, keys):
            separated_label[key].append(label)
            separated_prediction[key].append(prediction)
            if reduction == "mean":
                separated_weights[key] = 1
            elif reduction == "mean_by_sample_num":
                separated_weights[key] += 1
            elif reduction == "mean_by_positive_num":
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

    update_op = tf.py_func(update_pyfunc, [labels, predictions, gids], [])
    value_op = tf.py_func(value_pyfunc, [], tf.float32)
    return value_op, update_op


def pcopc(labels, predictions):
    separated_label = []
    separated_prediction = []

    def update_pyfunc(labels, predictions):
        for label, prediction in zip(labels, predictions):
            separated_label.append(label)
            separated_prediction.append(prediction)

    def value_pyfunc():
        return np.sum(separated_prediction) / np.sum(separated_label)

    update_op = tf.py_func(update_pyfunc, [labels, predictions], [])
    value_op = tf.py_func(value_pyfunc, [], tf.float32)
    return value_op, update_op


def normalized_discounted_cumulative_gain(labels, predictions, topn=None):
    """
    Computes normalized discounted cumulative gain (NDCG).
    Args:
        labels:
        predictions:
        topn:

    Returns:

    """
    pass
