# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/30 4:37 下午
# desc:

import os
import logging
from typing import List, Dict
from collections import OrderedDict
import tensorflow as tf
import easy_rec_ext.core.metrics as metrics_lib
from easy_rec_ext.model.multi_tower import MultiTower
from easy_rec_ext.layers import dnn
from easy_rec_ext.model.star import StarTopologyFCNLayer, AuxiliaryNetworkLayer as STARAuxiliaryNetworkLayer

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


class ESMMModelConfig(object):
    def __init__(self, label_names: List[str],
                 loss_weight_dict: Dict = None,
                 share_fn_param: bool = False, fn_param_dict=None
                 ):
        self.label_names = label_names
        self.share_fn_param = share_fn_param
        self.loss_weight_dict = loss_weight_dict
        self.fn_param_dict = fn_param_dict

    @staticmethod
    def handle(data):
        res = ESMMModelConfig(data["label_names"])
        if "share_fn_param" in data:
            res.share_fn_param = data["share_fn_param"]
        if "loss_weight_dict" in data:
            res.loss_weight_dict = data["loss_weight_dict"]
        if "fn_param_dict" in data:
            res.fn_param_dict = data["fn_param_dict"]
        return res

    def __str__(self):
        return str(self.__dict__)


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
        model_config = self._model_config.esmm_model_config

        task_tower_fea_arr_list = []
        task_all_fea_list = []
        if model_config.share_fn_param:
            tower_fea_arr = self.build_tower_fea_arr()
            all_fea = tf.concat(tower_fea_arr, axis=1)
            for i in range(len(model_config.label_names)):
                task_name = model_config.label_names[i]
                task_tower_fea_arr_list.append(tower_fea_arr)
                task_all_fea_list.append(all_fea)
                logging.info("%s build_predict_graph, task:%s, tower_fea_arr.length:%s" % (
                    filename, task_name, str(len(tower_fea_arr))))
                logging.info(
                    "%s build_predict_graph, task:%s, all_fea.shape:%s" % (
                        filename, task_name, str(all_fea.shape)))
        elif model_config.fn_param_dict:
            tower_outputs = dict()
            for i in range(len(model_config.label_names)):
                task_name = model_config.label_names[i]
                if task_name in model_config.fn_param_dict:
                    if model_config.fn_param_dict[task_name] in tower_outputs:
                        tower_fea_arr = tower_outputs[model_config.fn_param_dict[task_name]]
                    else:
                        tower_fea_arr = self.build_tower_fea_arr(variable_scope="task_%s" % model_config.fn_param_dict[task_name])
                        tower_outputs[model_config.fn_param_dict[task_name]] = tower_fea_arr
                else:
                    tower_fea_arr = self.build_tower_fea_arr(variable_scope="task_%s" % task_name)
                all_fea = tf.concat(tower_fea_arr, axis=1)
                task_tower_fea_arr_list.append(tower_fea_arr)
                task_all_fea_list.append(all_fea)
                logging.info("%s build_predict_graph, task:%s, tower_fea_arr.length:%s" % (
                    filename, task_name, str(len(tower_fea_arr))))
                logging.info(
                    "%s build_predict_graph, task:%s, all_fea.shape:%s" % (
                        filename, task_name, str(all_fea.shape)))
        else:
            for i in range(len(model_config.label_names)):
                task_name = model_config.label_names[i]
                tower_fea_arr = self.build_tower_fea_arr(variable_scope="task_%s" % task_name)
                all_fea = tf.concat(tower_fea_arr, axis=1)
                task_tower_fea_arr_list.append(tower_fea_arr)
                task_all_fea_list.append(all_fea)
                logging.info("%s build_predict_graph, task:%s, tower_fea_arr.length:%s" % (
                    filename, task_name, str(len(tower_fea_arr))))
                logging.info(
                    "%s build_predict_graph, task:%s, all_fea.shape:%s" % (
                        filename, task_name, str(all_fea.shape)))

        task_probs_list = []
        prediction_dict = OrderedDict()
        for i in range(len(model_config.label_names)):
            task_name = model_config.label_names[i]

            if self._model_config.star_model_config:
                star_model_config = self._model_config.star_model_config
                domain_id = self.get_id_feature(
                    star_model_config.domain_input_group, star_model_config.domain_id_col,
                    use_raw_id=True
                )
                task_logits = StarTopologyFCNLayer().call(
                    name=task_name + "_" + "star_fcn", deep_fea=task_all_fea_list[i], domain_id=domain_id,
                    domain_size=self._model_config.star_model_config.domain_size,
                    mlp_units=self._model_config.final_dnn.hidden_units,
                )
            else:
                task_logits = dnn.DNN(self._model_config.final_dnn, self._l2_reg,
                                      task_name + "_" + "final_dnn", self._is_training
                                      )(task_all_fea_list[i])

            if self._model_config.wide_towers:
                wide_fea = tf.concat(self._wide_tower_features, axis=1)
                task_logits = tf.concat([task_logits, wide_fea], axis=1)
                logging.info("%s build_predict_graph, task:%s, task_logits.shape:%s" % (
                    filename, task_name, str(task_logits.shape)))
            if self._model_config.bias_tower:
                bias_fea = self.build_bias_input_layer(self._model_config.bias_tower.input_group)
                task_logits = tf.concat([task_logits, bias_fea], axis=1)
                logging.info("%s build_predict_graph, task:%s, task_logits.shape:%s" % (
                    filename, task_name, str(task_logits.shape)))

            if self._model_config.star_model_config:
                star_model_config = self._model_config.star_model_config
                task_logits = STARAuxiliaryNetworkLayer().call(
                    name=task_name + "_" + "star_aux", deep_fea=task_logits,
                    domain_fea=self.get_id_feature(
                        star_model_config.domain_input_group, star_model_config.domain_id_col,
                        use_raw_id=False
                    ),
                    mlp_units=star_model_config.auxiliary_network_mlp_units,
                )
            else:
                task_logits = tf.layers.dense(task_logits, 1, name="%s_logits" % task_name)

            task_probs = tf.sigmoid(task_logits, name="%s_probs" % task_name)
            task_probs_list.append(task_probs)
            prediction_dict["%s_probs" % task_name] = tf.reshape(task_probs, (-1,))

        self._add_to_prediction_dict(prediction_dict)
        return self._prediction_dict

    def build_loss_graph(self):
        model_config = self._model_config.esmm_model_config

        prev_probs = None
        for i in range(len(model_config.label_names)):
            task_name = model_config.label_names[i]
            task_probs = self._prediction_dict["%s_probs" % task_name]
            if i > 0:
                real_probs = tf.multiply(prev_probs, task_probs)
                logloss = model_config.loss_weight_dict.get(task_name, 1.0) * tf.losses.log_loss(
                    labels=tf.cast(self._labels[task_name], tf.float32),
                    predictions=real_probs,
                )
                prev_probs = real_probs
            else:
                logloss = model_config.loss_weight_dict.get(task_name, 1.0) * tf.losses.log_loss(
                    labels=tf.cast(self._labels[task_name], tf.float32),
                    predictions=task_probs,
                )
                prev_probs = task_probs
            self._loss_dict["%s_cross_entropy_loss" % task_name] = logloss

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
        model_config = self._model_config.esmm_model_config
        metric_dict = OrderedDict()

        prev_probs = None
        for i in range(len(model_config.label_names)):
            task_name = model_config.label_names[i]
            task_probs = self._prediction_dict["%s_probs" % task_name]
            task_label = self._labels[task_name]

            if i > 0:
                pre_task_name = model_config.label_names[i - 1]
                task_mask = self._labels[pre_task_name] > 0
                task_label_mask = tf.boolean_mask(task_label, task_mask)
                task_probs_mask = tf.boolean_mask(task_probs, task_mask)

                for metric in eval_config.metric_set:
                    if "auc" == metric.name:
                        metric_dict["%s_auc_mask" % task_name] = tf.metrics.auc(
                            labels=tf.to_int64(task_label_mask),
                            predictions=task_probs_mask,
                        )
                    elif "gauc" == metric.name:
                        gids = tf.squeeze(self._feature_dict[metric.gid_field], axis=1)
                        metric_dict["%s_gauc_mask" % task_name] = metrics_lib.gauc(
                            labels=tf.to_int64(task_label_mask),
                            predictions=task_probs_mask,
                            gids=gids,
                            reduction=metric.reduction,
                        )
                    elif "pcopc" == metric.name:
                        metric_dict["%s_pcopc_mask" % task_name] = metrics_lib.pcopc(
                            labels=tf.to_float(task_label_mask),
                            predictions=task_probs_mask,
                        )

                task_probs = tf.multiply(prev_probs, task_probs)

            prev_probs = task_probs

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
