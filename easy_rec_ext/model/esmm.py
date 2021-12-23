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


class ESMMModelConfig(object):
    def __init__(self, ctr_label_name: str = "ctr_label", ctcvr_label_name: str = "ctcvr_label",
                 share_fn_param: int = 0,
                 ctr_loss_weight: float = 1.0, ctcvr_loss_weight: float = 1.0,
                 formula="dot", alpha: float = None,
                 ):
        self.ctr_label_name = ctr_label_name
        self.ctcvr_label_name = ctcvr_label_name
        self.share_fn_param = share_fn_param
        self.ctr_loss_weight = ctr_loss_weight
        self.ctcvr_loss_weight = ctcvr_loss_weight
        self.formula = formula
        self.alpha = alpha

    @staticmethod
    def handle(data):
        res = ESMMModelConfig()
        if "ctr_label_name" in data:
            res.ctr_label_name = data["ctr_label_name"]
        if "ctcvr_label_name" in data:
            res.ctcvr_label_name = data["ctcvr_label_name"]
        if "share_fn_param" in data:
            res.share_fn_param = data["share_fn_param"]
        if "ctr_loss_weight" in data:
            res.ctr_loss_weight = data["ctr_loss_weight"]
        if "ctcvr_loss_weight" in data:
            res.ctcvr_loss_weight = data["ctcvr_loss_weight"]
        if "formula" in data:
            res.formula = data["formula"]
        if "alpha" in data:
            res.alpha = data["alpha"]
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
        if self._model_config.esmm_model_config.share_fn_param == 1:
            tower_fea_arr = self.build_tower_fea_arr()
            ctr_tower_fea_arr = tower_fea_arr
            cvr_tower_fea_arr = tower_fea_arr
        else:
            ctr_tower_fea_arr = self.build_tower_fea_arr(variable_scope="task_ctr")
            cvr_tower_fea_arr = self.build_tower_fea_arr(variable_scope="task_cvr")

        logging.info("%s build_predict_graph, ctr_tower_fea_arr.length:%s" % (filename, str(len(ctr_tower_fea_arr))))
        ctr_all_fea = tf.concat(ctr_tower_fea_arr, axis=1)
        logging.info("%s build_predict_graph, ctr_all_fea.shape:%s" % (filename, str(ctr_all_fea.shape)))

        logging.info("%s build_predict_graph, cvr_tower_fea_arr.length:%s" % (filename, str(len(cvr_tower_fea_arr))))
        cvr_all_fea = tf.concat(cvr_tower_fea_arr, axis=1)
        logging.info("%s build_predict_graph, cvr_all_fea.shape:%s" % (filename, str(cvr_all_fea.shape)))

        ctr_final_dnn_layer = dnn.DNN(self._model_config.final_dnn, self._l2_reg,
                                      "ctr" + "_" + "final_dnn", self._is_training
                                      )
        ctr_logits = ctr_final_dnn_layer(ctr_all_fea)
        if self._model_config.wide_towers:
            wide_fea = tf.concat(self._wide_tower_features, axis=1)
            ctr_logits = tf.concat([ctr_logits, wide_fea], axis=1)
            logging.info("build_predict_graph, ctr_logits.shape:%s" % (str(ctr_logits.shape)))
        if self._model_config.bias_tower:
            bias_fea = self.build_bias_input_layer(self._model_config.bias_tower.input_group)
            ctr_logits = tf.concat([ctr_logits, bias_fea], axis=1)
            logging.info("build_predict_graph, ctr_logits.shape:%s" % (str(ctr_logits.shape)))
        ctr_logits = tf.layers.dense(ctr_logits, 1, name="ctr_logits")

        cvr_final_dnn_layer = dnn.DNN(self._model_config.final_dnn, self._l2_reg,
                                      "cvr" + "_" + "final_dnn", self._is_training
                                      )
        cvr_logits = cvr_final_dnn_layer(cvr_all_fea)
        if self._model_config.wide_towers:
            wide_fea = tf.concat(self._wide_tower_features, axis=1)
            cvr_logits = tf.concat([cvr_logits, wide_fea], axis=1)
            logging.info("build_predict_graph, cvr_logits.shape:%s" % (str(cvr_logits.shape)))
        if self._model_config.bias_tower:
            bias_fea = self.build_bias_input_layer(self._model_config.bias_tower.input_group)
            cvr_logits = tf.concat([cvr_logits, bias_fea], axis=1)
            logging.info("build_predict_graph, cvr_logits.shape:%s" % (str(cvr_logits.shape)))
        cvr_logits = tf.layers.dense(cvr_logits, 1, name="cvr_logits")

        ctr_probs = tf.sigmoid(ctr_logits, name="ctr_probs")
        cvr_probs = tf.sigmoid(cvr_logits, name="cvr_probs")

        if self._model_config.esmm_model_config.formula == "pow":
            if self._model_config.esmm_model_config.alpha:
                alpha = self._model_config.esmm_model_config.alpha
            else:
                with tf.variable_scope("esmm", reuse=tf.AUTO_REUSE):
                    alpha = tf.get_variable(name="alpha", shape=[1], dtype=tf.float32,
                                            initializer=tf.constant_initializer(0.5),
                                            )
            alpha = tf.clip_by_value(alpha, clip_value_min=0.0, clip_value_max=1.0)
            ctcvr_probs = tf.multiply(ctr_probs, tf.pow(cvr_probs, alpha), name="ctcvr_probs")
        else:
            ctcvr_probs = tf.multiply(ctr_probs, cvr_probs, name="ctcvr_probs")
        all_probs = tf.concat([ctr_probs, cvr_probs, ctcvr_probs], axis=-1, name="all_probs")

        prediction_dict = dict()
        if self._model_config.esmm_model_config.formula == "pow":
            prediction_dict["alpha"] = alpha
        prediction_dict["ctr_probs"] = tf.reshape(ctr_probs, (-1,))
        prediction_dict["cvr_probs"] = tf.reshape(cvr_probs, (-1,))
        prediction_dict["ctcvr_probs"] = tf.reshape(ctcvr_probs, (-1,))
        prediction_dict["all_probs"] = all_probs
        self._add_to_prediction_dict(prediction_dict)
        return self._prediction_dict

    def build_loss_graph(self):
        ctr_probs = self._prediction_dict["ctr_probs"]
        ctcvr_probs = self._prediction_dict["ctcvr_probs"]

        ctr_label = self._labels[self._model_config.esmm_model_config.ctr_label_name]
        ctcvr_label = self._labels[self._model_config.esmm_model_config.ctcvr_label_name]

        ctr_loss = self._model_config.esmm_model_config.ctr_loss_weight * tf.losses.log_loss(
            labels=tf.cast(ctr_label, tf.float32),
            predictions=ctr_probs,
        )
        ctcvr_loss = self._model_config.esmm_model_config.ctcvr_loss_weight * tf.losses.log_loss(
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

        ctr_label = self._labels[self._model_config.esmm_model_config.ctr_label_name]
        ctcvr_label = self._labels[self._model_config.esmm_model_config.ctcvr_label_name]

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
                metric_dict["ctr_pcopc"] = metrics_lib.pcopc(tf.to_float(ctr_label), ctr_probs)
                metric_dict["ctcvr_pcopc"] = metrics_lib.pcopc(tf.to_float(ctcvr_label),
                                                               ctcvr_probs)
        return metric_dict
