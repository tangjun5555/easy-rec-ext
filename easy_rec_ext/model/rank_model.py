# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/30 4:41 下午
# desc:

import logging
from abc import abstractmethod
from easy_rec_ext.core import embedding_ops
from easy_rec_ext.core import regularizers
import easy_rec_ext.core.metrics as metrics_lib

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
        self._model_config = model_config
        self._feature_config = feature_config

        self._feature_dict = features

        self._labels = labels
        if self._labels is not None:
            self._label_name = list(self._labels.keys())[0]

        self._is_training = is_training

        self._emb_reg = regularizers.l2_regularizer(self._model_config.embedding_regularization)
        self._l2_reg = regularizers.l2_regularizer(self._model_config.l2_regularization)

        self._prediction_dict = {}
        self._loss_dict = {}

        self._feature_groups_dict = {
            feature_group.group_name: feature_group
            for feature_group in self._model_config.feature_groups
        }
        self._feature_fields_dict = {
            feature_field.input_name: feature_field
            for feature_field in self._feature_config.feature_fields
        }

    def _add_to_prediction_dict(self, output):
        self._prediction_dict.update(output)

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

    def build_metric_graph(self, eval_config):
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

    def build_input_layer(self, feature_group):
        feature_group = self._feature_groups_dict[feature_group]
        outputs = []

        feature_fields_num = len(feature_group.feature_names) if feature_group.feature_names else 0
        for i in range(feature_fields_num):
            feature_field = self._feature_fields_dict[feature_group.feature_names[i]]
            if feature_field.feature_type == "IdFeature":
                embedding_weights = embedding_ops.get_embedding_variable(
                    feature_field.embedding_name,
                    feature_field.embedding_dim
                )
                values = embedding_ops.safe_embedding_lookup(
                    embedding_weights, self._feature_dict[feature_field.input_name],
                )
            elif feature_field.feature_type == "RawFeature":
                values = self._feature_dict[feature_field.input_name]
            elif feature_field.feature_type == "SequenceFeature":
                embedding_weights = embedding_ops.get_embedding_variable(
                    feature_field.embedding_name,
                    feature_field.embedding_dim
                )
                values = embedding_ops.safe_embedding_lookup(
                    embedding_weights, self._feature_dict[feature_field.input_name],
                    combiner=feature_field.combiner
                )
            else:
                continue
            outputs.append(values)
            logging.debug("build_input_layer, name:" + str(feature_field.input_name) + ", shape:" + str(
                values.get_shape().as_list()))
        outputs = tf.concat(outputs, axis=1)
        return outputs

    def build_seq_att_input_layer(self, feature_group):
        feature_group = self._feature_groups_dict[feature_group]
        outputs = {}

        key_feature_field = self._feature_fields_dict[feature_group.seq_att_map.key]
        assert key_feature_field.feature_type == "IdFeature"

        key_embedding_weights = embedding_ops.get_embedding_variable(
            key_feature_field.embedding_name,
            key_feature_field.embedding_dim
        )
        outputs["key"] = embedding_ops.safe_embedding_lookup(
            key_embedding_weights, self._feature_dict[key_feature_field.input_name],
        )

        seq_feature_field = self._feature_fields_dict[feature_group.seq_att_map.hist_seq]
        assert seq_feature_field.feature_type == "SequenceFeature"

        hist_seq = self._feature_dict[seq_feature_field.input_name]
        seq_embedding_weights = embedding_ops.get_embedding_variable(
            seq_feature_field.embedding_name,
            seq_feature_field.embedding_dim
        )
        hist_seq_emb = embedding_ops.safe_embedding_lookup(
            seq_embedding_weights, tf.expand_dims(hist_seq, -1)
        )
        outputs["hist_seq_emb"] = hist_seq_emb

        hist_seq_len = tf.where(tf.less(hist_seq, 0), tf.zeros_like(hist_seq), tf.ones_like(hist_seq))
        hist_seq_len = tf.reduce_sum(hist_seq_len, axis=1, keep_dims=False)
        outputs["hist_seq_len"] = hist_seq_len
        return outputs

    def build_bias_input_layer(self, feature_group):
        feature_group = self._feature_groups_dict[feature_group]
        outputs = []
        feature_fields_num = len(feature_group.feature_names) if feature_group.feature_names else 0
        for i in range(feature_fields_num):
            feature_field = self._feature_fields_dict[feature_group.feature_names[i]]
            assert feature_field.feature_type == "IdFeature"
            if feature_field.hash_bucket_size <= 0:
                outputs.append(
                    tf.one_hot(self._feature_dict[feature_field.input_name], feature_field.num_buckets)
                )
            else:
                outputs.append(
                    tf.one_hot(self._feature_dict[feature_field.input_name], feature_field.hash_bucket_size)
                )
        outputs = tf.concat(outputs, axis=-1)
        outputs = tf.squeeze(outputs)
        return outputs
