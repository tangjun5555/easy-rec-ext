# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/30 4:37 下午
# desc:

import os
import logging
import tensorflow as tf
from easy_rec_ext.model.multi_tower import MultiTower
from easy_rec_ext.layers import dnn
import easy_rec_ext.core.metrics as metrics_lib

if tf.__version__ >= "2.0":
    tf = tf.compat.v1

filename = str(os.path.basename(__file__)).split(".")[0]


class ESMM(MultiTower):
    def __init__(self,
                 model_config,
                 feature_config,
                 features,
                 labels=None,
                 is_training=False,
                 ):
        super(ESMM, self).__init__(model_config, feature_config, features, labels, is_training)

    def build_predict_graph(self):
        tower_fea_arr = self.build_tower_fea_arr()
        logging.info("%s build_predict_graph, tower_fea_arr.length:%s" % (filename, str(len(tower_fea_arr))))

        all_fea = tf.concat(tower_fea_arr, axis=1)
        logging.info("%s build_predict_graph, all_fea.shape:%s" % (filename, str(all_fea.shape)))

        ctr_final_dnn_layer = dnn.DNN(self._model_config.final_dnn, self._l2_reg,
                                      "ctr" + "_" + "final_dnn", self._is_training
                                      )
        ctr_logits = ctr_final_dnn_layer(all_fea)
        ctr_logits = tf.layers.dense(ctr_logits, 1, name="ctr_logits")

        cvr_final_dnn_layer = dnn.DNN(self._model_config.final_dnn, self._l2_reg,
                                      "cvr" + "_" + "final_dnn", self._is_training
                                      )
        cvr_logits = cvr_final_dnn_layer(all_fea)
        cvr_logits = tf.layers.dense(cvr_logits, 1, name="cvr_logits")

        ctr_probs = tf.sigmoid(ctr_logits, name="ctr_probs")
        cvr_probs = tf.sigmoid(cvr_logits, name="cvr_probs")
        ctcvr_probs = tf.multiply(ctr_probs, cvr_probs, name="ctcvr_probs")
        all_probs = tf.concat([ctr_probs, cvr_probs, ctcvr_probs], axis=-1, name="all_probs")

        prediction_dict = dict()
        prediction_dict["ctr_probs"] = tf.reshape(ctr_probs, (-1,))
        prediction_dict["cvr_probs"] = tf.reshape(cvr_probs, (-1,))
        prediction_dict["ctcvr_probs"] = tf.reshape(ctcvr_probs, (-1,))
        prediction_dict["all_probs"] = all_probs
        self._add_to_prediction_dict(prediction_dict)
        return self._prediction_dict

    def build_loss_graph(self):
        ctr_probs = self._prediction_dict["ctr_probs"]
        ctcvr_probs = self._prediction_dict["ctcvr_probs"]

        ctr_label = self._labels[self._model_config.ctcvr_label_config.ctr_label_name]
        ctcvr_label = self._labels[self._model_config.ctcvr_label_config.ctcvr_label_name]

        ctr_loss = self._model_config.ctcvr_label_config.ctr_loss_weight * tf.losses.log_loss(
            labels=tf.cast(ctr_label, tf.float32),
            predictions=ctr_probs,
        )
        ctcvr_loss = self._model_config.ctcvr_label_config.ctcvr_loss_weight * tf.losses.log_loss(
            labels=tf.cast(ctcvr_label, tf.float32),
            predictions=ctcvr_probs,
        )

        self._loss_dict["ctr_cross_entropy_loss"] = ctr_loss
        self._loss_dict["ctcvr_cross_entropy_loss"] = ctcvr_loss

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if regularization_losses:
            regularization_losses = [
                reg_loss.get() if hasattr(reg_loss, "get") else reg_loss
                for reg_loss in regularization_losses
            ]
            regularization_losses = tf.add_n(regularization_losses, name="regularization_loss")
            self._loss_dict["regularization_loss"] = regularization_losses

        return self._loss_dict

    def build_metric_graph(self, eval_config):
        ctr_probs = self._prediction_dict["ctr_probs"]
        ctcvr_probs = self._prediction_dict["ctcvr_probs"]

        ctr_label = self._labels[self._model_config.ctcvr_label_config.ctr_label_name]
        ctcvr_label = self._labels[self._model_config.ctcvr_label_config.ctcvr_label_name]

        metric_dict = {}
        for metric in eval_config.metric_set:
            if "auc" == metric.name:
                metric_dict["ctr_auc"] = tf.metrics.auc(tf.to_int64(ctr_label), ctr_probs)
                metric_dict["ctcvr_auc"] = tf.metrics.auc(tf.to_int64(ctcvr_label), ctcvr_probs)
            elif "gauc" == metric.name:
                gids = tf.squeeze(self._feature_dict[metric.gid_field], axis=1)
                metric_dict["ctr_gauc"] = metrics_lib.gauc(
                    tf.to_int64(ctr_label),
                    ctr_probs,
                    gids=gids,
                    reduction=metric.reduction
                )
                metric_dict["ctcvr_gauc"] = metrics_lib.gauc(
                    tf.to_int64(ctcvr_label),
                    ctcvr_probs,
                    gids=gids,
                    reduction=metric.reduction
                )
            elif "pcopc" == metric.name:
                metric_dict["ctr_pcopc"] = metrics_lib.pcopc(tf.to_float(ctr_label), self._prediction_dict["ctr_probs"])
                metric_dict["ctcvrpcopc"] = metrics_lib.pcopc(tf.to_float(ctcvr_label), self._prediction_dict["ctcvr_probs"])
        return metric_dict
