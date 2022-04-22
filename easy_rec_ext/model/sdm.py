# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/19 6:35 下午
# desc:
# Reference:
# SDM: Sequential Deep Matching Model for Online Large-scale Recommender System

import os
import logging
from typing import List
import tensorflow as tf
from easy_rec_ext.layers import dnn
from easy_rec_ext.layers.target_attention_layer import TargetAttention
from easy_rec_ext.layers.multihead_attention import MultiHeadSelfAttention
from easy_rec_ext.model.dssm import DSSMModel
from easy_rec_ext.model.match_model import MatchModel

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


class SDMModelConfig(object):
    def __init__(self, user_input_groups: List[str], hist_long_input_group: str,
                 hist_short_input_group: str, hist_short_seq_size: int,
                 item_input_groups: List[str],
                 user_field: str = None, item_field: str = None, scale_sim: bool = True,
                 ):
        self.user_input_groups = user_input_groups
        self.hist_long_input_group = hist_long_input_group
        self.hist_short_input_group = hist_short_input_group
        self.hist_short_seq_size = hist_short_seq_size
        self.item_input_groups = item_input_groups

        self.user_field = user_field
        self.item_field = item_field
        self.scale_sim = scale_sim

    @staticmethod
    def handle(data):
        res = SDMModelConfig(data["user_input_groups"], data["hist_long_input_group"],
                             data["hist_short_input_group"], data["hist_short_seq_size"],
                             data["item_input_groups"])
        if "user_field" in data:
            res.user_field = data["user_field"]
        if "item_field" in data:
            res.item_field = data["item_field"]
        if "scale_sim" in data:
            res.scale_sim = data["scale_sim"]
        return res


class SDMModel(DSSMModel):
    def sdm(self, name, output_units, hist_short_size, hist_long_feas, hist_short_feas, user_profile_emd):
        hist_long_seq_emb, hist_long_seq_len = hist_long_feas["hist_seq_emb"], hist_long_feas["hist_seq_len"]
        hist_short_seq_emb, hist_short_seq_len = hist_short_feas["hist_seq_emb"], hist_short_feas["hist_seq_len"]

        user_profile_emd = tf.concat(
            values=[
                user_profile_emd,
                tf.reduce_sum(hist_long_seq_emb, axis=1, keep_dims=False),
                tf.reduce_sum(hist_short_seq_emb, axis=1, keep_dims=False),
            ],
            axis=1
        )
        logging.info("%s %s, user_profile_emd.shape:%s" % (filename, name, str(user_profile_emd.shape)))
        user_profile_emd = tf.layers.dense(user_profile_emd, output_units, activation=tf.nn.tanh,
                                           name="%s_dense_1" % name)

        hist_long_seq_emb = tf.layers.dense(hist_long_seq_emb, output_units, activation=tf.nn.tanh,
                                            name="%s_dense_2" % name)
        prefer_output = TargetAttention(name="%s_prefer_target_att" % name, )(
            user_profile_emd, hist_long_seq_emb, hist_long_seq_len,
        )

        short_rnn_output = tf.keras.layers.LSTM(
            units=output_units,
            return_sequences=True,
            go_backwards=True,
            name="%s_lstm" % name,
        )(hist_short_seq_emb)
        short_att_output = MultiHeadSelfAttention(
            name="%s_self_att" % name,
            head_num=1,
            head_size=output_units,
            feature_num=hist_short_size,
        )(short_rnn_output, hist_short_seq_len)
        short_output = TargetAttention(name="%s_short_target_att" % name, )(
            user_profile_emd, short_att_output, hist_short_seq_len,
        )

        gate_input = tf.concat([prefer_output, short_output, user_profile_emd], axis=1)
        gate = tf.layers.dense(gate_input, output_units, activation=tf.nn.sigmoid, name="%s_gate" % name)
        gate_output = tf.multiply(gate, short_output) + tf.multiply(1 - gate, prefer_output)
        logging.info("%s %s, gate_output.shape:%s" % (filename, name, str(gate_output.shape)))
        return gate_output


class SDM(MatchModel, SDMModel):
    def __init__(self,
                 model_config,
                 feature_config,
                 features,
                 labels=None,
                 is_training=False,
                 ):
        super(SDM, self).__init__(model_config, feature_config, features, labels, is_training)

        self._dnn_tower_num = len(self._model_config.dnn_towers) if self._model_config.dnn_towers else 0
        self._dnn_tower_features = []
        for tower_id in range(self._dnn_tower_num):
            tower = self._model_config.dnn_towers[tower_id]
            tower_feature = self.build_input_layer(tower.input_group)
            self._dnn_tower_features.append(tower_feature)

        self._seq_pooling_tower_num = len(
            self._model_config.seq_pooling_towers) if self._model_config.seq_pooling_towers else 0
        self._seq_pooling_tower_names = []
        self._seq_pooling_tower_features = []
        for tower_id in range(self._seq_pooling_tower_num):
            tower = self._model_config.seq_pooling_towers[tower_id]
            self._seq_pooling_tower_names.append(tower.input_group)
            tower_feature = self.build_seq_input_layer(tower.input_group)
            self._seq_pooling_tower_features.append(tower_feature)

        logging.info("%s all tower num:%d" % (filename, self._dnn_tower_num + self._seq_pooling_tower_num))
        logging.info("%s dnn tower num:%d" % (filename, self._dnn_tower_num))
        logging.info("%s seq_pooling tower num:%d" % (filename, self._seq_pooling_tower_num))

    def build_predict_graph(self):
        sdm_model_config = self._model_config.sdm_model_config

        user_emb_fea_list = []
        item_emb_fea_list = []
        for tower_id in range(self._dnn_tower_num):
            tower = self._model_config.dnn_towers[tower_id]
            tower_fea = self._dnn_tower_features[tower_id]

            tower_fea = tf.layers.batch_normalization(
                tower_fea,
                training=self._is_training,
                trainable=True,
                name="%s_fea_bn" % tower.input_group,
            )
            tower_fea = dnn.DNN(
                tower.dnn_config,
                self._l2_reg,
                "%s_dnn" % tower.input_group,
                self._is_training
            )(tower_fea)

            if tower.input_group in sdm_model_config.user_input_groups:
                user_emb_fea_list.append(tower_fea)
            elif tower.input_group in sdm_model_config.item_input_groups:
                item_emb_fea_list.append(tower_fea)
            else:
                raise ValueError(
                    "%s tower.input_group:%s not in user or item input groups" % (filename, tower.input_group))
            logging.info("%s user_emb_fea_list add input_group:%s" % (filename, tower.input_group))
        user_emb_fea = tf.concat(user_emb_fea_list, axis=1)
        item_emb_fea = tf.concat(item_emb_fea_list, axis=1)

        item_emb = dnn.DNN(self._model_config.final_dnn,
                           self._l2_reg,
                           "item_final_dnn",
                           self._is_training
                           )(item_emb_fea)
        final_vector_size = self._model_config.final_dnn.hidden_units[-1]

        hist_long_feas = self._seq_pooling_tower_features[
            self._seq_pooling_tower_names.index(sdm_model_config.hist_long_input_group)
        ]
        hist_short_feas = self._seq_pooling_tower_features[
            self._seq_pooling_tower_names.index(sdm_model_config.hist_short_input_group)
        ]
        user_emb = self.sdm(
            "sdm", final_vector_size, sdm_model_config.hist_short_seq_size,
            hist_long_feas, hist_short_feas, user_emb_fea,
        )

        user_emb = self.norm(user_emb)
        item_emb = self.norm(item_emb)
        user_item_sim = self.sim(user_emb, item_emb)

        if sdm_model_config.scale_sim:
            sim_w = tf.get_variable(
                "sdm/scale_sim_w",
                dtype=tf.float32,
                shape=(1, 1),
                initializer=tf.ones_initializer(),
            )
            sim_b = tf.get_variable(
                "sdm/scale_sim_b",
                dtype=tf.float32,
                shape=(1,),
                initializer=tf.zeros_initializer()
            )
            tf.summary.histogram("sdm/scale_sim_w", sim_w)
            tf.summary.histogram("sdm/scale_sim_b", sim_b)
            user_item_sim = tf.matmul(user_item_sim, tf.abs(sim_w)) + sim_b
        probs = tf.nn.sigmoid(user_item_sim)

        self._prediction_dict["probs"] = tf.reshape(probs, (-1,), name="probs")

        self._prediction_dict["user_vector"] = tf.identity(user_emb, name="user_vector")
        self._prediction_dict["item_vector"] = tf.identity(item_emb, name="item_vector")

        if sdm_model_config.user_field:
            self._prediction_dict["user_id"] = tf.identity(self._feature_dict[sdm_model_config.user_field],
                                                           name="user_id")
        if sdm_model_config.item_field:
            self._prediction_dict["item_id"] = tf.identity(self._feature_dict[sdm_model_config.item_field],
                                                           name="item_id")
        return self._prediction_dict
