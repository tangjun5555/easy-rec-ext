# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/30 4:41 下午
# desc:

import os
import logging
from enum import Enum, unique
from abc import abstractmethod
from collections import OrderedDict
from easy_rec_ext.core import embedding_ops
from easy_rec_ext.core import regularizers
from easy_rec_ext.utils import constant
import easy_rec_ext.core.metrics as metrics_lib
from easy_rec_ext.layers.sequence_pooling import SequencePooling
from easy_rec_ext.layers.senet import SENetLayer

import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


@unique
class Similarity(Enum):
    COSINE = 0
    INNER_PRODUCT = 1
    EUCLID = 2

    @staticmethod
    def handle(value):
        if value == "COSINE":
            return Similarity.COSINE
        elif value == "INNER_PRODUCT":
            return Similarity.INNER_PRODUCT
        elif value == "EUCLID":
            return Similarity.EUCLID
        else:
            return None


class MatchModel(object):
    def __init__(self,
                 model_config,
                 feature_config,
                 features,
                 labels={},
                 is_training=False,
                 ):
        self._model_config = model_config
        self._feature_config = feature_config

        self._feature_dict = features

        self._labels = labels
        if self._labels is not None and len(self._labels.keys()) > 0:
            self._label_name = list(self._labels.keys())[0]

        # add sample weight from inputs
        self._sample_weight = 1.0
        if constant.SAMPLE_WEIGHT in features:
            self._sample_weight = features[constant.SAMPLE_WEIGHT]

        self._is_training = is_training

        self._emb_reg = regularizers.l2_regularizer(self._model_config.embedding_regularization)
        self._l2_reg = regularizers.l2_regularizer(self._model_config.l2_regularization)

        self._prediction_dict = OrderedDict()
        self._loss_dict = OrderedDict()

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

    def build_reg_loss(self):
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if regularization_losses:
            regularization_losses = [
                reg_loss.get() if hasattr(reg_loss, "get") else reg_loss
                for reg_loss in regularization_losses
            ]
            regularization_losses = tf.add_n(regularization_losses, name="regularization_loss")
            self._loss_dict["regularization_loss"] = regularization_losses

    def build_loss_graph(self):
        self.build_reg_loss()
        self._loss_dict["cross_entropy_loss"] = tf.losses.log_loss(
            labels=self._labels[self._label_name],
            predictions=self._prediction_dict["probs"],
            weights=tf.reshape(self._sample_weight, (-1,)),
        )
        return self._loss_dict

    def build_metric_graph(self, eval_config):
        metric_dict = {}
        for metric in eval_config.metric_set:
            label = tf.to_int64(self._labels[self._label_name])
            if "auc" == metric.name:
                metric_dict[str(metric)] = tf.metrics.auc(label, self._prediction_dict["probs"])
            elif "gauc" == metric.name:
                gids = tf.squeeze(self._feature_dict[metric.gid_field], axis=1)
                metric_dict[str(metric)] = metrics_lib.gauc(
                    label,
                    self._prediction_dict["probs"],
                    gids=gids,
                    reduction=metric.reduction,
                )
            elif "pcopc" == metric.name:
                label = tf.to_float(self._labels[self._label_name])
                metric_dict[str(metric)] = metrics_lib.pcopc(label, self._prediction_dict["probs"])
            elif "recall_at_k" == metric.name:
                metric_dict[str(metric)] = tf.metrics.recall_at_k(
                    label,
                    self._prediction_dict["probs"],
                    metric.topk,
                )
            else:
                raise NotImplemented
        return metric_dict

    def build_seq_att_input_layer(self, feature_group):
        logging.info("%s build_seq_att_input_layer, feature_group:%s" % (filename, str(feature_group)))
        outputs = {}
        group_input_dict = self.build_group_input_dict(feature_group)
        feature_group = self._feature_groups_dict[feature_group]
        key_outputs = []
        hist_seq_emb_outputs = []
        for seq_att_map in feature_group.seq_att_map_list:
            if seq_att_map.key:
                key_feature_field = self._feature_fields_dict[seq_att_map.key]
                if not seq_att_map.key_embed_prefix:
                    key_outputs.append(group_input_dict[key_feature_field.input_name])
                else:
                    input_ids = self._feature_dict[key_feature_field.input_name]
                    embed_name = seq_att_map.key_embed_prefix + key_feature_field.embedding_name
                    if input_ids.dtype == tf.dtypes.string:
                        embedding_weights = embedding_ops.get_embedding_variable(
                            name=embed_name,
                            dim=key_feature_field.embedding_dim,
                            vocab_size=key_feature_field.num_buckets if key_feature_field.num_buckets > 0 else key_feature_field.hash_bucket_size,
                            key_is_string=True,
                        )
                    else:
                        embedding_weights = embedding_ops.get_embedding_variable(
                            name=embed_name,
                            dim=key_feature_field.embedding_dim,
                            vocab_size=key_feature_field.num_buckets if key_feature_field.num_buckets > 0 else key_feature_field.hash_bucket_size,
                            key_is_string=False,
                        )
                    key_outputs.append(
                        embedding_ops.safe_embedding_lookup(
                            embedding_weights, input_ids
                        )
                    )

            if seq_att_map.hist_seq:
                seq_feature_field = self._feature_fields_dict[seq_att_map.hist_seq]
                if seq_feature_field.feature_type == "SequenceFeature":
                    hist_seq_emb_outputs.append(group_input_dict[seq_feature_field.input_name])

                    hist_seq = self._feature_dict[seq_feature_field.input_name]
                    if "hist_seq_len" not in outputs:
                        hist_seq_len = tf.where(tf.less(hist_seq, 0), tf.zeros_like(hist_seq), tf.ones_like(hist_seq))
                        hist_seq_len = tf.reduce_sum(hist_seq_len, axis=1, keep_dims=False)
                        outputs["hist_seq_len"] = hist_seq_len
                elif seq_feature_field.feature_type == "RawFeature":
                    if seq_feature_field.raw_input_embedding_type:
                        hist_seq_emb_outputs.append(
                            tf.reshape(group_input_dict[seq_feature_field.input_name],
                                       [-1, seq_feature_field.raw_input_dim, seq_feature_field.embedding_dim])
                        )
                    else:
                        hist_seq_emb_outputs.append(
                            tf.reshape(group_input_dict[seq_feature_field.input_name],
                                       [-1, seq_feature_field.raw_input_dim, 1])
                        )

        outputs["key"] = tf.concat(key_outputs, axis=-1)
        outputs["hist_seq_emb"] = tf.concat(hist_seq_emb_outputs, axis=-1)
        if feature_group.seq_att_projection_dim > 0:
            outputs["key"] = tf.layers.dense(
                outputs["key"], feature_group.seq_att_projection_dim,
                activation=None, use_bias=False,
                name=feature_group.group_name + "/" + "key_projection",
            )
            outputs["hist_seq_emb"] = tf.layers.dense(
                outputs["hist_seq_emb"], feature_group.seq_att_projection_dim,
                activation=None, use_bias=False,
                name=feature_group.group_name + "/" + "seq_projection"
            )
        return outputs

    def build_seq_input_layer(self, feature_group):
        logging.info("%s build_seq_input_layer, feature_group:%s" % (filename, str(feature_group)))
        outputs = {}
        hist_seq_emb_list = []

        group_input_dict = self.build_group_input_dict(feature_group)
        feature_group = self._feature_groups_dict[feature_group]

        feature_fields_num = len(feature_group.feature_name_list) if feature_group.feature_name_list else 0
        for i in range(feature_fields_num):
            feature_field = self._feature_fields_dict[feature_group.feature_name_list[i]]
            hist_seq_emb_list.append(group_input_dict[feature_field.feature_name])

            hist_seq_id = self._feature_dict[feature_field.input_name]
            if "hist_seq_len" not in outputs:
                hist_seq_len = tf.where(tf.less(hist_seq_id, 0), tf.zeros_like(hist_seq_id), tf.ones_like(hist_seq_id))
                hist_seq_len = tf.reduce_sum(hist_seq_len, axis=1, keep_dims=False)
                outputs["hist_seq_len"] = hist_seq_len

        outputs["hist_seq_emb"] = tf.concat(hist_seq_emb_list, axis=-1)
        return outputs

    def build_input_layer(self, feature_group):
        logging.info("%s build_input_layer, feature_group:%s" % (filename, str(feature_group)))
        outputs = []

        group_input_dict = self.build_group_input_dict(feature_group)
        feature_group = self._feature_groups_dict[feature_group]

        feature_fields_num = len(feature_group.feature_name_list) if feature_group.feature_name_list else 0
        for i in range(feature_fields_num):
            feature_field = self._feature_fields_dict[feature_group.feature_name_list[i]]
            outputs.append(group_input_dict[feature_field.feature_name])

        outputs = tf.concat(outputs, axis=1)
        if feature_group.senet_layer_config is not None:
            embedding_dim = outputs.get_shape().as_list()[-1] // feature_fields_num
            outputs = tf.reshape(outputs, (-1, feature_fields_num, embedding_dim))
            outputs = SENetLayer(
                name=feature_group.group_name + "_senet",
                reduction_ratio=feature_group.senet_layer_config.reduction_ratio,
            )(outputs)
            outputs = tf.reshape(outputs, (-1, feature_fields_num * embedding_dim))
        return outputs

    def build_group_input_dict(self, feature_group):
        feature_group = self._feature_groups_dict[feature_group]
        outputs = {}

        feature_fields_num = len(feature_group.feature_name_list) if feature_group.feature_name_list else 0
        for i in range(feature_fields_num):
            feature_field = self._feature_fields_dict[feature_group.feature_name_list[i]]

            if feature_field.feature_type == "IdFeature":
                input_ids = self._feature_dict[feature_field.input_name]
                if feature_field.one_hot == 1:
                    if feature_field.num_buckets > 0:
                        values = tf.one_hot(input_ids, feature_field.num_buckets)
                    else:
                        values = tf.one_hot(input_ids, feature_field.hash_bucket_size)
                    values = tf.squeeze(values, axis=[1])
                else:
                    embedding_weights = embedding_ops.get_embedding_variable(
                        name=feature_field.embedding_name,
                        dim=feature_field.embedding_dim,
                        vocab_size=feature_field.num_buckets if feature_field.num_buckets > 0 else feature_field.hash_bucket_size,
                        key_is_string=input_ids.dtype == tf.dtypes.string,
                    )
                    values = embedding_ops.safe_embedding_lookup(
                        embedding_weights, input_ids
                    )

            elif feature_field.feature_type == "RawFeature":
                values = self._feature_dict[feature_field.input_name]
                if feature_field.raw_input_embedding_type == "field_embedding":
                    embedding_weights = embedding_ops.get_embedding_variable(
                        name=feature_field.embedding_name,
                        dim=feature_field.embedding_dim,
                        vocab_size=feature_field.raw_input_dim,
                        key_is_string=False,
                    )
                    values = tf.multiply(tf.expand_dims(values, axis=-1), embedding_weights)
                    values = tf.reshape(values, [-1, feature_field.raw_input_dim * feature_field.embedding_dim])
                elif feature_field.raw_input_embedding_type == "mlp":
                    values = tf.layers.dense(
                        values, units=feature_field.raw_input_dim * feature_field.embedding_dim,
                        activation=tf.nn.relu, name=feature_field.embedding_name + "_dnn"
                    )
                elif feature_field.raw_input_embedding_type == "auto_dis":
                    from easy_rec_ext.layers.auto_discretization import AutoDis
                    values = AutoDis(
                        feature_field.feature_name + "_auto_dis",
                        feature_field.raw_input_dim,
                        feature_field.embedding_dim,
                    )(values)
                else:
                    raise NotImplemented

            elif feature_field.feature_type == "SequenceFeature":
                hist_seq = self._feature_dict[feature_field.input_name]
                if hist_seq.dtype == tf.dtypes.string:
                    hist_seq_len = tf.where(tf.math.logical_or(tf.equal(hist_seq, ""), tf.equal(hist_seq, "-1")),
                                            tf.zeros_like(tf.string_to_hash_bucket_fast(hist_seq, 10)),
                                            tf.ones_like(tf.string_to_hash_bucket_fast(hist_seq, 10)),
                                            )
                else:
                    hist_seq_len = tf.where(tf.less(hist_seq, 0), tf.zeros_like(hist_seq), tf.ones_like(hist_seq))
                hist_seq_len = tf.reduce_sum(hist_seq_len, axis=1, keep_dims=False)

                embedding_weights = embedding_ops.get_embedding_variable(
                    name=feature_field.embedding_name,
                    dim=feature_field.embedding_dim,
                    vocab_size=feature_field.num_buckets if feature_field.num_buckets > 0 else feature_field.hash_bucket_size,
                    key_is_string=hist_seq.dtype == tf.dtypes.string,
                )
                values = embedding_ops.safe_embedding_lookup(
                    embedding_weights, tf.expand_dims(hist_seq, -1)
                )

                if feature_field.sequence_pooling_config is not None:
                    values = SequencePooling(
                        name=feature_field.input_name + "_pooling",
                        mode=feature_field.sequence_pooling_config.mode,
                        gru_config=feature_field.sequence_pooling_config.gru_config,
                        lstm_config=feature_field.sequence_pooling_config.lstm_config,
                        self_att_config=feature_field.sequence_pooling_config.self_att_config,
                    )(values, hist_seq_len)

            else:
                raise ValueError("build_group_input_dict, feature_type: %s not supported." % feature_field.feature_type)

            outputs[feature_field.input_name] = values
            logging.info("build_group_input_dict, name:%s, shape:%s" %
                         (feature_field.input_name, str(values.get_shape().as_list()))
                         )

        return outputs
