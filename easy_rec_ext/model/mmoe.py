# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/10/20 2:52 下午
# desc:

import os
from typing import List
import logging
import tensorflow as tf
from easy_rec_ext.model.multi_tower import MultiTower
from easy_rec_ext.layers import dnn
import easy_rec_ext.core.metrics as metrics_lib

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


class MMoEModelCofing(object):
    def __init__(self, label_names: List[str],
                 num_expert: int,
                 expert_dnn_config: dnn.DNNConfig,
                 ):
        self.label_names = label_names
        self.num_task = len(label_names)
        self.num_expert = num_expert
        self.expert_dnn_config = expert_dnn_config

    @staticmethod
    def handle(data):
        expert_dnn_config = dnn.DNNConfig.handle(data["expert_dnn_config"])
        res = MMoEModelCofing(data["label_names"], data["num_expert"], expert_dnn_config)
        return res


class MMoE(MultiTower):
    def __init__(self,
                 model_config,
                 feature_config,
                 features,
                 labels=None,
                 is_training=False,
                 ):
        super(MMoE, self).__init__(model_config, feature_config, features, labels, is_training)

    def gate(self, unit, deep_fea, name):
        fea = tf.layers.dense(
            inputs=deep_fea,
            units=unit,
            kernel_regularizer=self._l2_reg,
            name="%s/gate_dnn" % name
        )
        fea = tf.nn.softmax(fea, axis=1)
        return fea

    def build_predict_graph(self):
        model_config = self._model_config.mmoe_model_config

        tower_fea_arr = self.build_tower_fea_arr()
        logging.info("%s build_predict_graph, tower_fea_arr.length:%s" % (filename, str(len(tower_fea_arr))))

        all_fea = tf.concat(tower_fea_arr, axis=1)
        logging.info("%s build_predict_graph, all_fea.shape:%s" % (filename, str(all_fea.shape)))

        expert_fea_list = []
        for expert_id in range(model_config.num_expert):
            expert_dnn_config = model_config.expert_dnn_config
            expert_dnn = dnn.DNN(
                expert_dnn_config,
                self._l2_reg,
                name="mmoe/expert_%d_dnn" % expert_id,
                is_training=self._is_training
            )
            expert_fea = expert_dnn(all_fea)
            expert_fea_list.append(expert_fea)
        experts_fea = tf.stack(expert_fea_list, axis=1)

        task_input_list = []
        for task_id in range(model_config.num_task):
            gate = self.gate(
                unit=model_config.num_expert,
                deep_fea=all_fea,
                name="mmoe/gate_%d" % task_id
            )
            gate = tf.expand_dims(gate, -1)
            tf.summary.histogram("mmoe/gate_%d_weights" % task_id, gate)
            task_input = tf.multiply(experts_fea, gate)
            task_input = tf.reduce_sum(task_input, axis=1)
            task_input_list.append(task_input)

        prediction_dict = {}
        for i in range(model_config.num_task):
            task_name = model_config.label_names[i]
            task_output = dnn.DNN(self._model_config.final_dnn,
                                  self._l2_reg,
                                  "%s_final_dnn" % task_name,
                                  self._is_training,
                                  )(task_input_list[i])

            if self._model_config.wide_towers:
                wide_fea = tf.concat(self._wide_tower_features, axis=1)
                task_output = tf.concat([task_output, wide_fea], axis=1)
                logging.info("%s build_predict_graph, task:%s, task_output.shape:%s" % (
                filename, task_name, str(task_output.shape)))
            if self._model_config.bias_tower:
                bias_fea = self.build_bias_input_layer(self._model_config.bias_tower.input_group)
                task_output = tf.concat([task_output, bias_fea], axis=1)
                logging.info("%s build_predict_graph, task:%s, task_output.shape:%s" % (
                filename, task_name, str(task_output.shape)))

            task_output = tf.layers.dense(task_output, 1, name="%s_logits" % task_name)
            task_output = tf.sigmoid(task_output, name="%s_probs" % task_name)
            prediction_dict["%s_probs" % task_name] = tf.reshape(task_output, (-1,))
        self._add_to_prediction_dict(prediction_dict)
        return self._prediction_dict

    def build_loss_graph(self):
        model_config = self._model_config.mmoe_model_config
        for i in range(model_config.num_task):
            task_name = model_config.label_names[i]
            task_label = self._labels[task_name]
            task_output = self._prediction_dict["%s_probs" % task_name]
            task_loss = tf.losses.log_loss(
                labels=tf.cast(task_label, tf.float32),
                predictions=task_output,
            )
            self._loss_dict["%s_cross_entropy_loss" % task_name] = task_loss

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
        model_config = self._model_config.mmoe_model_config
        metric_dict = dict()
        for i in range(model_config.num_task):
            task_name = model_config.label_names[i]
            task_label = self._labels[task_name]
            task_output = self._prediction_dict["%s_probs" % task_name]
            for metric in eval_config.metric_set:
                if "auc" == metric.name:
                    metric_dict["%s_auc" % task_name] = tf.metrics.auc(tf.to_int64(task_label), task_output)
                elif "gauc" == metric.name:
                    gids = tf.squeeze(self._feature_dict[metric.gid_field], axis=1)
                    metric_dict["%s_gauc" % task_name] = metrics_lib.gauc(
                        tf.to_int64(task_label),
                        task_output,
                        gids=gids,
                        reduction=metric.reduction
                    )
                elif "pcopc" == metric.name:
                    metric_dict["%s_pcopc" % task_name] = metrics_lib.pcopc(tf.to_float(task_label), task_output)
        return metric_dict
