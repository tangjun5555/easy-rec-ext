# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/30 4:41 下午
# desc:

from abc import abstractmethod
import easy_rec_ext.core.metrics as metrics_lib
from easy_rec_ext.core.pipeline import EvalConfig

import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


class RankModel(object):
    def __init__(self,
                 model_config,
                 feature_config,
                 features,
                 labels=None,
                 is_training=False,
                 ):
        self._feature_dict = features

        self._labels = labels
        if self._labels is not None:
            self._label_name = list(self._labels.keys())[0]

        self._prediction_dict = {}
        self._loss_dict = {}

    @abstractmethod
    def build_predict_graph(self):
        return self._prediction_dict

    def build_loss_graph(self):
        self._loss_dict["cross_entropy_loss"] = tf.losses.log_loss(
            labels=self._labels[self._label_name],
            predictions=self._prediction_dict["probs"]
        )

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if regularization_losses:
            regularization_losses = [
                reg_loss.get() if hasattr(reg_loss, "get") else reg_loss
                for reg_loss in regularization_losses
            ]
            regularization_losses = tf.add_n(regularization_losses, name="regularization_loss")
            self._loss_dict["regularization_loss"] = regularization_losses

        return self._loss_dict

    def build_metric_graph(self, eval_config: EvalConfig):
        metric_dict = {}
        for metric in eval_config.metric_set:
            if "auc" == metric.name:
                label = tf.to_int64(self._labels[self._label_name])
                metric_dict["auc"] = tf.metrics.auc(label, self._prediction_dict["probs"])
            elif "gauc" == metric.name:
                label = tf.to_int64(self._labels[self._label_name])
                metric_dict["gauc"] = metrics_lib.gauc(
                    label,
                    self._prediction_dict["probs"],
                    uids=self._feature_dict[metric.gid_field],
                    reduction=metric.reduction
                )
            elif "pcopc" == metric.name:
                label = tf.to_float(self._labels[self._label_name])
                metric_dict["pcopc"] = metrics_lib.pcopc(label, self._prediction_dict["probs"])
        return metric_dict
