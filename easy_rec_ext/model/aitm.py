# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/30 4:37 下午
# desc:


import logging
import tensorflow as tf
from easy_rec_ext.layers import dnn
from easy_rec_ext.utils import variable_util
from easy_rec_ext.model.rank_model import RankModel
import easy_rec_ext.core.metrics as metrics_lib

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


class AITM(RankModel):
    def __init__(self,
                 model_config,
                 feature_config,
                 features,
                 labels=None,
                 is_training=False,
                 ):
        super(AITM, self).__init__(model_config, feature_config, features, labels, is_training)

        assert self._model_config.aitm_model is not None

        all_fea = []
        for feature_group in self._model_config.feature_groups:
            all_fea.append(
                self.build_input_layer(feature_group)
            )
        self.all_fea = tf.concat(all_fea, axis=1)

    def attention(self, input1, input2):
        # (N,L,K)
        inputs = tf.concat([input1[:, None, :], input2[:, None, :]], axis=1)

        attention_k = self._model_config.aitm_model.attention_k

        # (N,L,K)*(K,K)->(N,L,K), L=2, K=32 in this.
        attention_w1 = variable_util.get_normal_variable("AITM", "w1", [attention_k, attention_k])
        attention_w2 = variable_util.get_normal_variable("AITM", "w2", [attention_k, attention_k])
        attention_w3 = variable_util.get_normal_variable("AITM", "w3", [attention_k, attention_k])
        Q = tf.tensordot(inputs, attention_w1, axes=1)
        K = tf.tensordot(inputs, attention_w2, axes=1)
        V = tf.tensordot(inputs, attention_w3, axes=1)

        # (N,L)
        a = tf.reduce_sum(tf.multiply(Q, K), axis=-1) / tf.sqrt(tf.cast(inputs.shape[-1], tf.float32))
        a = tf.nn.softmax(a, axis=1)

        # (N,L,K)
        outputs = tf.multiply(a[:, :, None], V)
        return tf.reduce_sum(outputs, axis=1)  # (N, K)

    def build_predict_graph(self):
        input_net = self.all_fea
        logging.info("AITM build_predict_graph, input_net.shape:%s" % (str(input_net.shape)))

        raw_ctr = dnn.DNN(
            self._model_config.aitm_model.ctr_dnn_config,
            self._l2_reg, "%s_dnn" % "ctr", self._is_training
        )(input_net)
        raw_ctcvr = dnn.DNN(
            self._model_config.aitm_model.ctcvr_dnn_config,
            self._l2_reg, "%s_dnn" % "ctcvr", self._is_training
        )(input_net)
        logging.info("AITM build_predict_graph, raw_ctr.shape:%s" % (str(raw_ctr.shape)))
        logging.info("AITM build_predict_graph, raw_ctcvr.shape:%s" % (str(raw_ctcvr.shape)))

        raw_ctr = tf.layers.dense(raw_ctr, self._model_config.aitm_model.attention_k, activation=tf.nn.relu)
        raw_ctcvr = tf.layers.dense(raw_ctcvr, self._model_config.aitm_model.attention_k, activation=tf.nn.relu)

        info = tf.layers.dense(raw_ctr, self._model_config.aitm_model.attention_k, activation=tf.nn.relu)
        raw_ctcvr = self.attention(info, raw_ctcvr)

        ctr_probs = tf.sigmoid(tf.layers.dense(raw_ctr, 1))
        ctcvr_probs = tf.sigmoid(tf.layers.dense(raw_ctcvr, 1))

        prediction_dict = dict()
        prediction_dict["ctr_probs"] = ctr_probs
        prediction_dict["ctcvr_probs"] = ctcvr_probs
        self._add_to_prediction_dict(prediction_dict)
        return self._prediction_dict

    def build_loss_graph(self):
        ctr_probs = self._prediction_dict["ctr_probs"]
        ctcvr_probs = self._prediction_dict["ctcvr_probs"]

        ctr_label = self._labels[self._model_config.aitm_model.ctr_label_name]
        ctcvr_label = self._labels[self._model_config.aitm_model.ctcvr_label_name]

        ctr_loss = self._model_config.aitm_model.ctr_loss_weight * tf.losses.log_loss(
            labels=tf.cast(ctr_label, tf.float32),
            predictions=ctr_probs,
        )
        ctcvr_loss = self._model_config.aitm_model.ctcvr_loss_weight * tf.losses.log_loss(
            labels=tf.cast(ctcvr_label, tf.float32),
            predictions=ctcvr_probs,
        )
        label_constraint_loss = self._model_config.aitm_model.label_constraint_loss_weight * tf.reduce_mean(
            tf.maximum(ctcvr_probs - ctr_probs, tf.zeros_like(ctr_probs)), axis=0
        )

        self._loss_dict["ctr_cross_entropy_loss"] = ctr_loss
        self._loss_dict["ctcvr_cross_entropy_loss"] = ctcvr_loss
        self._loss_dict["label_constraint_loss"] = label_constraint_loss

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

        ctr_label = self._labels[self._model_config.aitm_model.ctr_label_name]
        ctcvr_label = self._labels[self._model_config.aitm_model.ctcvr_label_name]

        metric_dict = {}
        for metric in eval_config.metric_set:
            if "auc" == metric.name:
                metric_dict["ctr_auc"] = tf.metrics.auc(ctr_label, ctr_probs)
                metric_dict["ctcvr_auc"] = tf.metrics.auc(ctcvr_label, ctcvr_probs)
            elif "gauc" == metric.name:
                metric_dict["ctr_gauc"] = metrics_lib.gauc(
                    ctr_label,
                    ctr_probs,
                    uids=self._feature_dict[metric.gid_field],
                    reduction=metric.reduction
                )
                metric_dict["ctcvr_gauc"] = metrics_lib.gauc(
                    ctcvr_label,
                    ctcvr_probs,
                    uids=self._feature_dict[metric.gid_field],
                    reduction=metric.reduction
                )
            elif "pcopc" == metric.name:
                metric_dict["ctr_pcopc"] = metrics_lib.pcopc(ctr_label, self._prediction_dict["ctr_probs"])
                metric_dict["ctcvrpcopc"] = metrics_lib.pcopc(ctcvr_label, self._prediction_dict["ctcvr_probs"])
        return metric_dict
