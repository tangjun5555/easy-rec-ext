# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2022/4/25 4:12 PM
# desc:
# Reference:
# DropoutNet: Addressing Cold Start in Recommender Systems

import os
import logging
from typing import List
import tensorflow as tf
from easy_rec_ext.layers import dnn, sequence_pooling
from easy_rec_ext.model.dssm import DSSMModel
from easy_rec_ext.model.match_model import MatchModel

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


class DropoutNetModelConfig(object):
    def __init__(self, user_content_groups: List[str], item_content_groups: List[str],
                 user_preference_groups: List[str], item_preference_groups: List[str],
                 user_preference_dnn_confg: dnn.DNNConfig, item_preference_dnn_confg: dnn.DNNConfig,
                 user_dropout_rate: float, item_dropout_rate: float,
                 user_field: str = None, item_field: str = None,
                 scale_sim: bool = True,
                 ):
        self.user_content_groups = user_content_groups
        self.item_content_groups = item_content_groups

        self.user_preference_groups = user_preference_groups
        self.item_preference_groups = item_preference_groups

        self.user_preference_dnn_confg = user_preference_dnn_confg
        self.item_preference_dnn_confg = item_preference_dnn_confg

        self.user_dropout_rate = user_dropout_rate
        self.item_dropout_rate = item_dropout_rate

        self.user_field = user_field
        self.item_field = item_field
        self.scale_sim = scale_sim

    @staticmethod
    def handle(data):
        user_preference_dnn_confg = dnn.DNNConfig.handle(data["user_preference_dnn_confg"])
        item_preference_dnn_confg = dnn.DNNConfig.handle(data["item_preference_dnn_confg"])
        res = DropoutNetModelConfig(
            data["user_content_groups"], data["item_content_groups"],
            data["user_preference_groups"], data["item_preference_groups"],
            user_preference_dnn_confg, item_preference_dnn_confg,
            data["user_dropout_rate"], data["item_dropout_rate"],
        )
        if "user_field" in data:
            res.user_field = data["user_field"]
        if "item_field" in data:
            res.item_field = data["item_field"]
        if "scale_sim" in data:
            res.scale_sim = data["scale_sim"]
        return res


class DropoutNetModel(DSSMModel):
    def dropoutnet(self, name, preference_feature, dnn_config, dropout_rate, is_training):
        logging.info("%s %s, preference_feature.shape:%s" % (filename, name, str(preference_feature.shape)))
        if is_training:
            prob = tf.random.uniform(shape=[tf.shape(preference_feature)[0]])
            preference_feature = tf.where(
                tf.less(prob, dropout_rate),
                tf.zeros_like(preference_feature),
                preference_feature,
            )
        preference_feature = dnn.DNN(
            dnn_config=dnn_config,
            l2_reg=None,
            name="%s_dnn" % name,
            is_training=is_training,
        )(preference_feature)
        return preference_feature


class DropoutNet(MatchModel, DropoutNetModel):
    def __init__(self,
                 model_config,
                 feature_config,
                 features,
                 labels=None,
                 is_training=False,
                 ):
        super(DropoutNet, self).__init__(model_config, feature_config, features, labels, is_training)

        self._dnn_tower_num = len(self._model_config.dnn_towers) if self._model_config.dnn_towers else 0
        self._dnn_tower_features = []
        for tower_id in range(self._dnn_tower_num):
            tower = self._model_config.dnn_towers[tower_id]
            tower_feature = self.build_input_layer(tower.input_group)
            self._dnn_tower_features.append(tower_feature)

        self._seq_pooling_tower_num = len(
            self._model_config.seq_pooling_towers) if self._model_config.seq_pooling_towers else 0
        self._seq_pooling_tower_features = []
        for tower_id in range(self._seq_pooling_tower_num):
            tower = self._model_config.seq_pooling_towers[tower_id]
            tower_feature = self.build_seq_input_layer(tower.input_group)
            self._seq_pooling_tower_features.append(tower_feature)

        logging.info("%s all tower num:%d" % (filename, self._dnn_tower_num + self._seq_pooling_tower_num))
        logging.info("%s dnn tower num:%d" % (filename, self._dnn_tower_num))
        logging.info("%s seq_pooling tower num:%d" % (filename, self._seq_pooling_tower_num))

    def build_predict_graph(self):
        model_config = self._model_config.dropoutnet_model_config

        user_content_fea_list = []
        user_prefer_fea_list = []
        item_content_fea_list = []
        item_prefer_fea_list = []

        for tower_id in range(self._dnn_tower_num):
            tower = self._model_config.dnn_towers[tower_id]
            tower_fea = self._dnn_tower_features[tower_id]

            if tower.input_group in model_config.user_preference_groups:
                user_prefer_fea_list.append(tower_fea)
            elif tower.input_group in model_config.item_preference_groups:
                item_prefer_fea_list.append(tower_fea)
            else:
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

                if tower.input_group in model_config.user_content_groups:
                    user_content_fea_list.append(tower_fea)
                elif tower.input_group in model_config.item_content_groups:
                    item_content_fea_list.append(tower_fea)
                else:
                    raise ValueError("tower.input_group:%s not in user or item input groups" % tower.input_group)

        for tower_id in range(self._seq_pooling_tower_num):
            tower = self._model_config.seq_pooling_towers[tower_id]
            tower_fea = self._seq_pooling_tower_features[tower_id]

            if tower.input_group in model_config.user_preference_groups:
                user_prefer_fea_list.append(tower_fea)
            elif tower.input_group in model_config.item_preference_groups:
                item_prefer_fea_list.append(tower_fea)
            else:
                tower_fea = sequence_pooling.SequencePooling(
                    name=tower.input_group + "_pooling",
                    mode=tower.sequence_pooling_config.mode,
                    gru_config=tower.sequence_pooling_config.gru_config,
                    lstm_config=tower.sequence_pooling_config.lstm_config,
                    self_att_config=tower.sequence_pooling_config.self_att_config,
                )(tower_fea["hist_seq_emb"], tower_fea["hist_seq_len"])

                if tower.input_group in model_config.user_content_groups:
                    user_content_fea_list.append(tower_fea)
                elif tower.input_group in model_config.item_content_groups:
                    item_content_fea_list.append(tower_fea)
                else:
                    raise ValueError("tower.input_group:%s not in user or item input groups" % tower.input_group)

        user_content_fea = tf.concat(user_content_fea_list, axis=1)
        user_prefer_fea = tf.concat(user_prefer_fea_list, axis=1)
        item_content_fea = tf.concat(item_content_fea_list, axis=1)
        item_prefer_fea = tf.concat(item_prefer_fea_list, axis=1)

        user_prefer_fea = self.dropoutnet(
            "dropoutnet_user",
            user_prefer_fea,
            model_config.user_preference_dnn_confg,
            model_config.user_dropout_rate,
            self._is_training
        )
        item_prefer_fea = self.dropoutnet(
            "dropoutnet_item",
            item_prefer_fea,
            model_config.item_preference_dnn_confg,
            model_config.item_dropout_rate,
            self._is_training
        )

        user_emb_fea = tf.concat([user_content_fea, user_prefer_fea], axis=1)
        item_emb_fea = tf.concat([item_content_fea, item_prefer_fea], axis=1)

        user_emb = dnn.DNN(self._model_config.final_dnn,
                           self._l2_reg,
                           "user_final_dnn",
                           self._is_training
                           )(user_emb_fea)
        item_emb = dnn.DNN(self._model_config.final_dnn,
                           self._l2_reg,
                           "item_final_dnn",
                           self._is_training
                           )(item_emb_fea)

        # process bias tower
        if self._model_config.bias_towers:
            for tower in self._model_config.bias_towers:
                bias_fea = self.build_input_layer(tower.input_group)
                bias_fea = tf.layers.dense(bias_fea, user_emb.shape()[-1],
                                           name="bias_tower_dense_" + tower.input_group)
                if "multiply" == tower.fusion_mode:
                    user_emb = tf.multiply(user_emb, bias_fea)
                else:
                    user_emb = tf.add(user_emb, bias_fea)

        user_emb = self.norm(user_emb)
        item_emb = self.norm(item_emb)
        user_item_sim = self.sim(user_emb, item_emb)

        if model_config.scale_sim:
            sim_w = tf.get_variable(
                "dssm/scale_sim_w",
                dtype=tf.float32,
                shape=(1, 1),
                initializer=tf.ones_initializer(),
            )
            sim_b = tf.get_variable(
                "dropoutnet/scale_sim_b",
                dtype=tf.float32,
                shape=(1,),
                initializer=tf.zeros_initializer()
            )
            tf.summary.histogram("dropoutnet/scale_sim_w", sim_w)
            tf.summary.histogram("dropoutnet/scale_sim_b", sim_b)
            user_item_sim = tf.matmul(user_item_sim, tf.abs(sim_w)) + sim_b
        probs = tf.nn.sigmoid(user_item_sim)

        self._prediction_dict["probs"] = tf.reshape(probs, (-1,), name="probs")

        self._prediction_dict["user_vector"] = tf.identity(user_emb, name="user_vector")
        self._prediction_dict["item_vector"] = tf.identity(item_emb, name="item_vector")

        if model_config.user_field:
            self._prediction_dict["user_id"] = tf.identity(self._feature_dict[model_config.user_field],
                                                           name="user_id")
        if model_config.item_field:
            self._prediction_dict["item_id"] = tf.identity(self._feature_dict[model_config.item_field],
                                                           name="item_id")

        return self._prediction_dict
