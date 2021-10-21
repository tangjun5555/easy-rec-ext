# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/10/20 2:52 下午
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
        tower_fea_arr = self.build_tower_fea_arr()
        logging.info("%s build_predict_graph, tower_fea_arr.length:%s" % (filename, str(len(tower_fea_arr))))

        all_fea = tf.concat(tower_fea_arr, axis=1)
        logging.info("%s build_predict_graph, all_fea.shape:%s" % (filename, str(all_fea.shape)))

        expert_fea_list = []
        for expert_id in range(self._model_config.mmoe_model_config.num_expert):
            expert_dnn_config = self._model_config.mmoe_model_config.expert_dnn_config
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
        for task_id in range(self._model_config.mmoe_model_config.num_task):
            gate = self.gate(
                unit=self._model_config.mmoe_model_config.num_expert,
                deep_fea=experts_fea,
                name="mmoe/gate_%d" % task_id
            )
            gate = tf.expand_dims(gate, -1)
            task_input = tf.multiply(experts_fea, gate)
            task_input = tf.reduce_sum(task_input, axis=1)
            task_input_list.append(task_input)

        tower_outputs = {}
        tower_outputs_list = []
        for i in range(self._model_config.mmoe_model_config.num_task):
            tower_name = self._model_config.mmoe_model_config.label_names[i]
            tower_dnn = dnn.DNN(self._model_config.final_dnn,
                                self._l2_reg,
                                "%s_final_dnn" % tower_name,
                                self._is_training,
                                )
            tower_output = tower_dnn(task_input_list[i])
            tower_output = tf.layers.dense(tower_output, 1, name="%s_logits" % tower_name)
            tower_output = tf.sigmoid(tower_output, name="%s_probs" % tower_name)
            tower_outputs[tower_name] = tf.reshape(tower_output, (-1,))
            tower_outputs_list.append(tower_output)
        tower_outputs["all_probs"] = tf.concat(tower_outputs_list, axis=-1, name="all_probs")
        self._add_to_prediction_dict(tower_outputs)
        return self._prediction_dict

    def build_loss_graph(self):
        for i in range(self._model_config.mmoe_model_config.num_task):
            tower_name = self._model_config.mmoe_model_config.label_names[i]
            tower_label = self._labels[tower_name]
            tower_loss = tf.losses.log_loss(
                labels=tf.cast(tower_label, tf.float32),
                predictions=self._prediction_dict[tower_name],
            )
            self._loss_dict["%s_cross_entropy_loss" % tower_name] = tower_loss

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
        metric_dict = dict()
        for i in range(self._model_config.mmoe_model_config.num_task):
            tower_name = self._model_config.mmoe_model_config.label_names[i]
            tower_label = self._labels[tower_name]
            tower_output = self._prediction_dict[tower_name]
            for metric in eval_config.metric_set:
                if "auc" == metric.name:
                    metric_dict["%s_auc" % tower_name] = tf.metrics.auc(tf.to_int64(tower_label), tower_output)
                elif "gauc" == metric.name:
                    gids = tf.squeeze(self._feature_dict[metric.gid_field], axis=1)
                    metric_dict["%s_gauc" % tower_name] = metrics_lib.gauc(
                        tf.to_int64(tower_label),
                        tower_output,
                        gids=gids,
                        reduction=metric.reduction
                    )
                elif "pcopc" == metric.name:
                    metric_dict["%s_pcopc" % tower_name] = metrics_lib.pcopc(tf.to_float(tower_label), tower_output)
        return metric_dict
