# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/30 4:41 下午
# desc:

import logging
from abc import abstractmethod
from easy_rec_ext.core import embedding_ops
from easy_rec_ext.core import regularizers
import easy_rec_ext.core.metrics as metrics_lib
from easy_rec_ext.layers.sequence_pooling import SequencePooling

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

    def get_prediction_keys(self):
        return list(self._prediction_dict.keys())

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
                gids = tf.squeeze(self._feature_dict[metric.gid_field], axis=1)
                metric_dict["gauc"] = metrics_lib.gauc(
                    label,
                    self._prediction_dict["probs"],
                    gids=gids,
                    reduction=metric.reduction
                )
            elif "pcopc" == metric.name:
                label = tf.to_float(self._labels[self._label_name])
                metric_dict["pcopc"] = metrics_lib.pcopc(label, self._prediction_dict["probs"])
        return metric_dict

    def build_input_layer(self, feature_group):
        group_input_dict = self.build_group_input_dict(feature_group)
        outputs = []

        feature_group = self._feature_groups_dict[feature_group]
        feature_fields_num = len(feature_group.feature_names) if feature_group.feature_names else 0

        for i in range(feature_fields_num):
            feature_field = self._feature_fields_dict[feature_group.feature_names[i]]
            outputs.append(group_input_dict[feature_field.input_name])

        outputs = tf.concat(outputs, axis=1)
        return outputs

    def build_seq_att_input_layer(self, feature_group):
        group_input_dict = self.build_group_input_dict(feature_group)
        outputs = {}

        feature_group = self._feature_groups_dict[feature_group]
        logging.info("build_seq_att_input_layer, feature_group:%s" % str(feature_group))

        key_outputs = []
        hist_seq_emb_outputs = []
        for seq_att_map in feature_group.seq_att_map_list:

            key_feature_field = self._feature_fields_dict[seq_att_map.key]
            assert key_feature_field.feature_type == "IdFeature"
            key_outputs.append(group_input_dict[key_feature_field.input_name])

            seq_feature_field = self._feature_fields_dict[seq_att_map.hist_seq]
            assert seq_feature_field.feature_type == "SequenceFeature"
            hist_seq_emb_outputs.append(group_input_dict[seq_feature_field.input_name])

            hist_seq = self._feature_dict[seq_feature_field.input_name]

            if "hist_seq_len" not in outputs:
                hist_seq_len = tf.where(tf.less(hist_seq, 0), tf.zeros_like(hist_seq), tf.ones_like(hist_seq))
                hist_seq_len = tf.reduce_sum(hist_seq_len, axis=1, keep_dims=False)
                outputs["hist_seq_len"] = hist_seq_len

        outputs["key"] = tf.concat(key_outputs, axis=-1)
        outputs["hist_seq_emb"] = tf.concat(hist_seq_emb_outputs, axis=-1)
        return outputs

    def build_bias_input_layer(self, feature_group):
        return self.build_input_layer(feature_group)

    def build_interaction_input_layer(self, feature_group):
        group_input_dict = self.build_group_input_dict(feature_group)
        outputs = []

        feature_group = self._feature_groups_dict[feature_group]
        feature_fields_num = len(feature_group.feature_names) if feature_group.feature_names else 0

        for i in range(feature_fields_num):
            feature_field = self._feature_fields_dict[feature_group.feature_names[i]]
            if feature_field.feature_type == "IdFeature":
                outputs.append(group_input_dict[feature_field.input_name])
            elif feature_field.feature_type == "RawFeature":
                outputs.append(
                    tf.reshape(group_input_dict[feature_field.input_name], [-1, feature_field.raw_input_dim, feature_field.embedding_dim])
                )
            else:
                raise ValueError("build_interaction_input_layer, feature_type: %s not supported." % feature_field.feature_type)

        outputs = tf.stack(outputs, axis=1)
        return outputs

    def build_group_input_dict(self, feature_group):
        feature_group = self._feature_groups_dict[feature_group]
        outputs = {}

        feature_fields_num = len(feature_group.feature_names) if feature_group.feature_names else 0
        for i in range(feature_fields_num):
            feature_field = self._feature_fields_dict[feature_group.feature_names[i]]

            if feature_field.feature_type == "IdFeature":
                input_ids = self._feature_dict[feature_field.input_name]

                if feature_field.one_hot == 1:
                    if feature_field.num_buckets > 0:
                        values = tf.one_hot(input_ids, feature_field.num_buckets)
                    else:
                        values = tf.one_hot(input_ids, feature_field.hash_bucket_size)
                    values = tf.squeeze(values, axis=[1])

                else:
                    if input_ids.dtype == tf.dtypes.string:
                        embedding_weights = embedding_ops.get_embedding_variable(
                            name=feature_field.embedding_name,
                            dim=feature_field.embedding_dim,
                            vocab_size=feature_field.num_buckets if feature_field.num_buckets > 0 else feature_field.hash_bucket_size,
                            key_is_string=True,
                        )
                    else:
                        embedding_weights = embedding_ops.get_embedding_variable(
                            name=feature_field.embedding_name,
                            dim=feature_field.embedding_dim,
                            vocab_size=feature_field.num_buckets if feature_field.num_buckets > 0 else feature_field.hash_bucket_size,
                            key_is_string=False,
                        )
                    values = embedding_ops.safe_embedding_lookup(
                        embedding_weights, input_ids
                    )

            elif feature_field.feature_type == "RawFeature":
                values = self._feature_dict[feature_field.input_name]
                if feature_field.raw_input_use_field_embedding == 1:
                    embedding_weights = embedding_ops.get_embedding_variable(
                        name=feature_field.embedding_name,
                        dim=feature_field.embedding_dim,
                        vocab_size=feature_field.raw_input_dim,
                        key_is_string=False,
                    )
                    values = tf.multiply(tf.expand_dims(values, axis=-1), embedding_weights)
                    values = tf.reshape(tf.reshape(values, [-1, feature_field.raw_input_dim * feature_field.embedding_dim]))

            elif feature_field.feature_type == "SequenceFeature":
                hist_seq = self._feature_dict[feature_field.input_name]
                hist_seq_len = tf.where(tf.less(hist_seq, 0), tf.zeros_like(hist_seq), tf.ones_like(hist_seq))
                hist_seq_len = tf.reduce_sum(hist_seq_len, axis=1, keep_dims=False)

                embedding_weights = embedding_ops.get_embedding_variable(
                    name=feature_field.embedding_name,
                    dim=feature_field.embedding_dim,
                    vocab_size=feature_field.num_buckets if feature_field.num_buckets > 0 else feature_field.hash_bucket_size,
                    key_is_string=False,
                )
                values = embedding_ops.safe_embedding_lookup(
                    embedding_weights, tf.expand_dims(hist_seq, -1)
                )

                if feature_field.sequence_pooling_config is not None:
                    values = SequencePooling(
                        name=feature_field.input_name + "_pooling",
                        mode=feature_field.sequence_pooling_config.mode,
                        gru_config=feature_field.sequence_pooling_config.gru_config,
                    )(values, hist_seq_len)

            else:
                raise ValueError("build_group_input_dict, feature_type: %s not supported." % feature_field.feature_type)

            outputs[feature_field.input_name] = values
            logging.info("build_group_input_dict, name:%s, shape:%s" %
                         (feature_field.input_name, str(values.get_shape().as_list()))
                         )

        return outputs
