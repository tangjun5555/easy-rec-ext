# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/10/22 11:06 上午
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


class PLE(MultiTower):
    def __init__(self,
                 model_config,
                 feature_config,
                 features,
                 labels=None,
                 is_training=False,
                 ):
        super(PLE, self).__init__(model_config, feature_config, features, labels, is_training)

    def gate(self, selector_fea, vec_feas, name):
        vec = tf.stack(vec_feas, axis=1)
        gate = tf.layers.dense(
            inputs=selector_fea,
            units=len(vec_feas),
            kernel_regularizer=self._l2_reg,
            activation=None,
            name=name + "_gate/dnn"
        )
        gate = tf.nn.softmax(gate, axis=1)
        gate = tf.expand_dims(gate, -1)
        task_input = tf.multiply(vec, gate)
        task_input = tf.reduce_sum(task_input, axis=1)
        return task_input

    def experts_layer(self, deep_fea, expert_num, experts_cfg, name):
        tower_outputs = []
        for expert_id in range(expert_num):
            tower_dnn = dnn.DNN(
                experts_cfg,
                self._l2_reg,
                name=name + "_expert_%d/dnn" % expert_id,
                is_training=self._is_training
            )
            tower_output = tower_dnn(deep_fea)
            tower_outputs.append(tower_output)
        return tower_outputs

    def CGC_layer(self, task_num, num_expert_share, num_expert_per_task, 
                  layer_name, expert_dnn_config, 
                  extraction_network_fea, shared_expert_fea, 
                  final_flag):        
        expert_shared_out = self.experts_layer(
            shared_expert_fea, num_expert_share,
            expert_dnn_config, layer_name + "_share/dnn"
        )

        experts_outs = []
        cgc_layer_outs = []
        for task_idx in range(task_num):
            name = layer_name + "_task_%d" % task_idx
            experts_out = self.experts_layer(
                extraction_network_fea[task_idx], num_expert_per_task,
                expert_dnn_config, name
            )
            cgc_layer_out = self.gate(extraction_network_fea[task_idx],
                                      experts_out + expert_shared_out, name
                                      )
            experts_outs.extend(experts_out)
            cgc_layer_outs.append(cgc_layer_out)

        if final_flag:
            shared_layer_out = None
        else:
            shared_layer_out = self.gate(shared_expert_fea,
                                         experts_outs + expert_shared_out,
                                         layer_name + "_share"
                                         )
        return cgc_layer_outs, shared_layer_out

    def build_predict_graph(self):
        tower_fea_arr = self.build_tower_fea_arr()
        logging.info("%s build_predict_graph, tower_fea_arr.length:%s" % (filename, str(len(tower_fea_arr))))

        tower_fea_arr = tf.concat(tower_fea_arr, axis=1)
        logging.info("%s build_predict_graph, tower_fea_arr.shape:%s" % (filename, str(tower_fea_arr.shape)))

        shared_expert_fea = tower_fea_arr
        extraction_network_fea = [tower_fea_arr] * self._model_config.ple_model_config.num_task

        final_flag = False
        for idx in range(len(self._model_config.extraction_networks)):
            if idx == len(self._model_config.extraction_networks) - 1:
                final_flag = True
            extraction_network_fea, shared_expert_fea = self.CGC_layer(
                self._model_config.ple_model_config.num_task, 
                self._model_config.ple_model_config.num_expert_share, 
                self._model_config.ple_model_config.num_expert_per_task, 
                self._model_config.ple_model_config.label_names[idx], 
                self._model_config.ple_model_config.expert_dnn_config, 
                extraction_network_fea, shared_expert_fea,
                final_flag
            )
        
        tower_outputs = {}
        tower_outputs_list = []
        for i in range(self._model_config.ple_model_config.num_task):
            tower_name = self._model_config.ple_model_config.label_names[i]
            tower_dnn = dnn.DNN(self._model_config.final_dnn,
                                self._l2_reg,
                                "%s_final_dnn" % tower_name,
                                self._is_training,
                                )
            tower_output = tower_dnn(extraction_network_fea[i])
            tower_output = tf.layers.dense(tower_output, 1, name="%s_logits" % tower_name)
            tower_output = tf.sigmoid(tower_output, name="%s_probs" % tower_name)
            tower_outputs[tower_name] = tf.reshape(tower_output, (-1,))
            tower_outputs_list.append(tower_output)
        tower_outputs["all_probs"] = tf.concat(tower_outputs_list, axis=-1, name="all_probs")
        self._add_to_prediction_dict(tower_outputs)
        return self._prediction_dict

    def build_loss_graph(self):
        for i in range(self._model_config.ple_model_config.num_task):
            tower_name = self._model_config.ple_model_config.label_names[i]
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
        for i in range(self._model_config.ple_model_config.num_task):
            tower_name = self._model_config.ple_model_config.label_names[i]
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
