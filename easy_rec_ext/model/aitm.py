# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/12/13 4:47 PM
# desc:

import os
import logging
from typing import List, Dict
from easy_rec_ext.model.multi_tower import MultiTower
from easy_rec_ext.layers import dnn
import easy_rec_ext.core.metrics as metrics_lib
from easy_rec_ext.utils import variable_util

import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1

filename = str(os.path.basename(__file__)).split(".")[0]


class AITMModelConfig(object):
    def __init__(self, label_names: List[str],
                 share_fn_param: int = 0,
                 attention_dim: int = 64,
                 loss_weight_dict: Dict = None,
                 label_constraint_loss_weight: float = 1.0,
                 ):
        self.label_names = label_names
        self.share_fn_param = share_fn_param
        self.attention_dim = attention_dim
        self.loss_weight_dict = loss_weight_dict
        self.label_constraint_loss_weight = label_constraint_loss_weight

    @staticmethod
    def handle(data):
        res = AITMModelConfig(data["label_names"])
        if "share_fn_param" in data:
            res.share_fn_param = data["share_fn_param"]
        if "attention_dim" in data:
            res.attention_dim = data["attention_dim"]
        if "loss_weight_dict" in data:
            res.loss_weight_dict = data["loss_weight_dict"]
        if "label_constraint_loss_weight" in data:
            res.label_constraint_loss_weight = data["label_constraint_loss_weight"]
        return res

    def __str__(self):
        return str(self.__dict__)


class AITM(MultiTower):
    def __init__(self,
                 model_config,
                 feature_config,
                 features,
                 labels=None,
                 is_training=False,
                 ):
        super(AITM, self).__init__(model_config, feature_config, features, labels, is_training)

    def attention(self, input1, input2, attention_dim, name):
        # (N,L,K)
        inputs = tf.concat([input1[:, None, :], input2[:, None, :]], axis=1)

        # (N,L,K)*(K,K)->(N,L,K), L=2, K=32 in this.
        attention_w1 = variable_util.get_normal_variable("AITM", "%s_w1" % name, [attention_dim, attention_dim])
        attention_w2 = variable_util.get_normal_variable("AITM", "%s_w2" % name, [attention_dim, attention_dim])
        attention_w3 = variable_util.get_normal_variable("AITM", "%s_w3" % name, [attention_dim, attention_dim])
        Q = tf.tensordot(inputs, attention_w1, axes=1)
        K = tf.tensordot(inputs, attention_w2, axes=1)
        V = tf.tensordot(inputs, attention_w3, axes=1)

        # (N,L)
        a = tf.reduce_sum(tf.multiply(Q, K), axis=-1) / tf.sqrt(tf.cast(inputs.shape[-1], tf.float32))
        a = tf.nn.softmax(a, axis=1)
        tf.summary.histogram("aitm/%s_attention_score" % name, a)

        # (N,L,K)
        outputs = tf.multiply(a[:, :, None], V)
        return tf.reduce_sum(outputs, axis=1)  # (N, K)

    def build_predict_graph(self):
        task_tower_fea_arr = []
        task_all_fea = []
        if self._model_config.aitm_model_config.share_fn_param == 1:
            tower_fea_arr = self.build_tower_fea_arr()
            all_fea = tf.concat(tower_fea_arr, axis=1)
            for i in range(len(self._model_config.aitm_model_config.label_names)):
                task_name = self._model_config.aitm_model_config.label_names[i]
                task_tower_fea_arr.append(tower_fea_arr)
                task_all_fea.append(all_fea)
                logging.info("%s build_predict_graph, task:%s, tower_fea_arr.length:%s" % (
                    filename, task_name, str(len(tower_fea_arr))))
                logging.info(
                    "%s build_predict_graph, task:%s, all_fea.shape:%s" % (filename, task_name, str(all_fea.shape)))
        else:
            for task_name in self._model_config.aitm_model_config.label_names:
                tower_fea_arr = self.build_tower_fea_arr(variable_scope="task_%s" % task_name)
                all_fea = tf.concat(tower_fea_arr, axis=1)
                task_tower_fea_arr.append(tower_fea_arr)
                task_all_fea.append(all_fea)
                logging.info("%s build_predict_graph, task:%s, tower_fea_arr.length:%s" % (
                    filename, task_name, str(len(tower_fea_arr))))
                logging.info(
                    "%s build_predict_graph, task:%s, all_fea.shape:%s" % (filename, task_name, str(all_fea.shape)))

        attention_dim = self._model_config.aitm_model_config.attention_dim
        raw_logits_list = [None]
        prediction_dict = dict()
        for i in range(len(self._model_config.aitm_model_config.label_names)):
            task_name = self._model_config.aitm_model_config.label_names[i]
            task_raw_logits = dnn.DNN(self._model_config.final_dnn,
                                      self._l2_reg,
                                      task_name + "_" + "final_dnn",
                                      self._is_training,
                                      )(task_all_fea[i])
            if self._model_config.wide_towers:
                wide_fea = tf.concat(self._wide_tower_features, axis=1)
                task_raw_logits = tf.concat([task_raw_logits, wide_fea], axis=1)
                logging.info("%s build_predict_graph, task:%s, task_raw_logits.shape:%s" % (filename, task_name, str(task_raw_logits.shape)))
            task_raw_logits = tf.layers.dense(task_raw_logits,
                                              attention_dim,
                                              activation=tf.nn.relu,
                                              name=task_name + "_" + "dnn_last",
                                              )
            raw_logits_list.append(task_raw_logits)

            if i > 0:
                info = tf.layers.dense(raw_logits_list[i], attention_dim, activation=tf.nn.relu)
                task_logits = self.attention(task_raw_logits, info, attention_dim, task_name)
            else:
                task_logits = task_raw_logits

            if self._model_config.bias_tower:
                bias_fea = self.build_bias_input_layer(self._model_config.bias_tower.input_group)
                task_logits = tf.concat([task_logits, bias_fea], axis=1)
                logging.info("%s build_predict_graph, task:%s, task_logits.shape:%s" % (filename, task_name, str(task_logits.shape)))

            task_logits = tf.layers.dense(task_logits, 1, name="%s_logits" % task_name)
            task_probs = tf.sigmoid(task_logits, name="%s_probs" % task_name)
            prediction_dict["%s_probs" % task_name] = tf.reshape(task_probs, (-1,))
        self._add_to_prediction_dict(prediction_dict)
        return self._prediction_dict

    def build_loss_graph(self):
        for i in range(len(self._model_config.aitm_model_config.label_names)):
            task_name = self._model_config.aitm_model_config.label_names[i]
            task_probs = self._prediction_dict["%s_probs" % task_name]
            logloss = self._model_config.aitm_model_config.loss_weight_dict.get(task_name, 1.0) * tf.losses.log_loss(
                labels=tf.cast(self._labels[task_name], tf.float32),
                predictions=task_probs,
            )
            self._loss_dict["%s_cross_entropy_loss" % task_name] = logloss

            if i > 0:
                prev_task_name = self._model_config.aitm_model_config.label_names[i - 1]
                label_constraint_loss = self._model_config.aitm_model_config.label_constraint_loss_weight * tf.reduce_mean(
                    tf.maximum(task_probs - self._prediction_dict["%s_probs" % prev_task_name],
                               tf.zeros_like(task_probs)), axis=0
                )
                self._loss_dict["%s_%s_label_constraint_loss" % (prev_task_name, task_name)] = label_constraint_loss

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
        metric_dict = {}

        for i in range(len(self._model_config.aitm_model_config.label_names)):
            task_name = self._model_config.aitm_model_config.label_names[i]
            task_probs = self._prediction_dict["%s_probs" % task_name]
            task_label = self._labels[task_name]

            for metric in eval_config.metric_set:
                if "auc" == metric.name:
                    metric_dict["%s_auc" % task_name] = tf.metrics.auc(tf.to_int64(task_label), task_probs)
                elif "gauc" == metric.name:
                    gids = tf.squeeze(self._feature_dict[metric.gid_field], axis=1)
                    metric_dict["%s_gauc" % task_name] = metrics_lib.gauc(
                        tf.to_int64(task_label), task_probs,
                        gids=gids, reduction=metric.reduction
                    )
                elif "pcopc" == metric.name:
                    metric_dict["%s_pcopc" % task_name] = metrics_lib.pcopc(tf.to_float(task_label), task_probs)
        return metric_dict
