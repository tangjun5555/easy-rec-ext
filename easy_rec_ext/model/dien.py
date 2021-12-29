# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/12/23 5:07 PM
# desc:

import os
import logging
import tensorflow as tf
from easy_rec_ext.layers import dnn
from easy_rec_ext.core import regularizers
from easy_rec_ext.model.rank_model import RankModel

if tf.__version__ >= "2.0":
    tf = tf.compat.v1

filename = str(os.path.basename(__file__)).split(".")[0]


class DIENConfig(object):
    def __init__(self, use_auxiliary_loss: int = 0, combine_mechanism: str = "AIGRU", return_target: bool = True):
        self.use_auxiliary_loss = use_auxiliary_loss
        self.combine_mechanism = combine_mechanism
        self.return_target = return_target

    @staticmethod
    def handle(data):
        res = DIENConfig()
        if "use_auxiliary_loss" in data:
            res.use_auxiliary_loss = data["use_auxiliary_loss"]
        if "combine_mechanism" in data:
            res.combine_mechanism = data["combine_mechanism"]
        if "return_target" in data:
            res.return_target = data["return_target"]
        return res

    def __str__(self):
        return str(self.__dict__)


class DIENTower(object):
    def __init__(self, input_group, dien_config: DIENConfig):
        self.input_group = input_group
        self.dien_config = dien_config

    @staticmethod
    def handle(data):
        res = DIENTower(data["input_group"], DIENConfig.handle(data["dien_config"]))
        return res

    def __str__(self):
        return str(self.__dict__)


class DIENLayer(object):
    def dien(self, name, deep_fea, combine_mechanism, return_target=True):
        cur_id, hist_id_col, seq_len = deep_fea["key"], deep_fea["hist_seq_emb"], deep_fea["hist_seq_len"]
        seq_max_len = hist_id_col.get_shape().as_list()[1]
        emb_dim = hist_id_col.get_shape().as_list()[2]

        hist_gru = self.interest_extractor(name, emb_dim, hist_id_col)
        final_state = self.interest_evolving(name, cur_id, seq_len, seq_max_len, emb_dim, hist_gru, combine_mechanism)

        if return_target:
            dien_output = tf.concat([final_state, cur_id], axis=1)
        else:
            dien_output = final_state
        logging.info("%s %s, dien_output.shape:%s" % (filename, name, str(dien_output.shape)))
        return dien_output

    def auxiliary_loss(self):
        """
        TODO
        Returns:

        """
        return None

    def interest_extractor(self, name, emb_dim, hist_id_col):
        hist_gru = tf.keras.layers.GRU(
            units=emb_dim,
            return_sequences=True,
            name="%s_interest_extractor_gru" % name,
        )(hist_id_col)
        return hist_gru

    def AIGRU(self, name, seq_max_len, emb_dim, hist_gru, hist_attention):
        final_state = tf.math.multiply(hist_gru, tf.reshape(hist_attention, (-1, seq_max_len, 1)))
        final_state = tf.keras.layers.GRU(
            units=emb_dim,
            return_sequences=False,
            name="%s_AIGRU" % name,
        )(final_state)
        return final_state

    def AGRU(self):
        """
        TODO
        Returns:

        """
        pass

    def AUGRU(self):
        """
        TODO
        Returns:

        """
        pass

    def interest_evolving(self, name, cur_id, seq_len, seq_max_len, emb_dim, hist_gru, combine_mechanism):
        # scaled Dot-Product
        hist_attention = tf.matmul(tf.expand_dims(cur_id, 1), hist_gru, transpose_b=True) / (emb_dim ** -0.5)

        # mask
        seq_len = tf.expand_dims(seq_len, 1)
        mask = tf.sequence_mask(seq_len, seq_max_len)
        padding = tf.ones_like(hist_attention) * (-2 ** 32 + 1)
        hist_attention = tf.where(mask, hist_attention, padding)

        hist_attention = tf.nn.softmax(hist_attention)
        tf.summary.histogram("%s_interest_evolving_hist_attention" % name, hist_attention)

        return self.AIGRU(name, seq_max_len, emb_dim, hist_gru, hist_attention)


class DIEN(RankModel, DIENLayer):
    def __init__(self,
                 model_config,
                 feature_config,
                 features,
                 labels=None,
                 is_training=False,
                 ):
        super(DIEN, self).__init__(model_config, feature_config, features, labels, is_training)

        self._dnn_tower_num = len(self._model_config.dnn_towers) if self._model_config.dnn_towers else 0
        self._dnn_tower_features = []
        for tower_id in range(self._dnn_tower_num):
            tower = self._model_config.dnn_towers[tower_id]
            tower_feature = self.build_input_layer(tower.input_group)
            self._dnn_tower_features.append(tower_feature)

        self._dien_tower_num = len(self._model_config.dien_towers)
        self._dien_tower_features = []
        for tower_id in range(self._dien_tower_num):
            tower = self._model_config.dien_towers[tower_id]
            tower_feature = self.build_seq_att_input_layer(tower.input_group)
            regularizers.apply_regularization(self._emb_reg, weights_list=[tower_feature["key"]])
            regularizers.apply_regularization(self._emb_reg, weights_list=[tower_feature["hist_seq_emb"]])
            self._dien_tower_features.append(tower_feature)

        logging.info("all tower num: {0}".format(self._dnn_tower_num + self._dien_tower_num))
        logging.info("dnn tower num: {0}".format(self._dnn_tower_num))
        logging.info("dien tower num: {0}".format(self._dien_tower_num))

    def build_predict_graph(self):
        tower_fea_arr = []

        for tower_id in range(self._dnn_tower_num):
            tower_fea = self._dnn_tower_features[tower_id]
            tower = self._model_config.dnn_towers[tower_id]
            tower_name = tower.input_group
            tower_fea = tf.layers.batch_normalization(
                tower_fea,
                training=self._is_training,
                trainable=True,
                name="%s_fea_bn" % tower_name,
            )
            dnn_layer = dnn.DNN(
                tower.dnn_config,
                self._l2_reg,
                "%s_dnn" % tower_name,
                self._is_training,
            )
            tower_fea = dnn_layer(tower_fea)
            tower_fea_arr.append(tower_fea)

        for tower_id in range(self._dien_tower_num):
            tower_fea = self._dien_tower_features[tower_id]
            tower = self._model_config.dien_towers[tower_id]
            tower_name = tower.input_group
            tower_fea = self.dien(
                name="%s_dien" % tower_name,
                deep_fea=tower_fea,
                combine_mechanism=tower.dien_config.combine_mechanism,
                return_target=tower.dien_config.return_target,
            )
            tower_fea_arr.append(tower_fea)

        all_fea = tf.concat(tower_fea_arr, axis=1)
        final_dnn_layer = dnn.DNN(self._model_config.final_dnn, self._l2_reg, "final_dnn", self._is_training)
        all_fea = final_dnn_layer(all_fea)
        logging.info("%s, build_predict_graph, all_fea.shape:%s" % (filename, str(all_fea.shape)))

        if self._model_config.bias_tower:
            bias_fea = self.build_bias_input_layer(self._model_config.bias_tower.input_group)
            all_fea = tf.concat([all_fea, bias_fea], axis=1)
            logging.info("%s, build_predict_graph, all_fea.shape:%s" % (filename, str(all_fea.shape)))

        logits = tf.layers.dense(all_fea, 1, name="logits")
        logits = tf.reshape(logits, (-1,))
        probs = tf.sigmoid(logits, name="probs")

        prediction_dict = dict()
        prediction_dict["probs"] = probs
        self._add_to_prediction_dict(prediction_dict)
        return self._prediction_dict
