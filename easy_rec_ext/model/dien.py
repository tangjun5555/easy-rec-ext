# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/12/23 5:07 PM
# desc: Deep Interest Evolution Network

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
    def __init__(self, seq_size: int,
                 combine_mechanism: str = "AIGRU",
                 return_target: bool = True,
                 ):
        self.seq_size = seq_size
        self.combine_mechanism = combine_mechanism
        self.return_target = return_target

    @staticmethod
    def handle(data):
        res = DIENConfig(data["seq_size"])
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
    def dien(self, name, deep_fea, seq_size, combine_mechanism, return_target=True):
        cur_id, hist_id_col, seq_len = deep_fea["key"], deep_fea["hist_seq_emb"], deep_fea["hist_seq_len"]
        emb_dim = hist_id_col.get_shape().as_list()[2]

        hist_gru = self.interest_extractor_layer(name, emb_dim, hist_id_col)
        final_state = self.interest_evolving_layer(name, cur_id, seq_len, seq_size, emb_dim, hist_gru, combine_mechanism)

        if return_target:
            dien_output = tf.concat([final_state, cur_id], axis=1)
        else:
            dien_output = final_state
        logging.info("%s %s, dien_output.shape:%s" % (filename, name, str(dien_output.shape)))
        return dien_output

    def interest_extractor_layer(self, name, emb_dim, hist_id_col):
        hist_gru = tf.keras.layers.GRU(
            units=emb_dim,
            return_sequences=True,
            go_backwards=True,
            name="%s_interest_extractor_gru" % name,
        )(hist_id_col)
        logging.info("%s interest_extractor, hist_id_col.shape:%s, hist_gru.shape:%s" % (
            filename, str(hist_id_col.shape), str(hist_gru.shape)))
        return hist_gru

    def auxiliary_loss_layer(self):
        """

        Returns:

        """
        return None

    def interest_evolving_layer(self, name, cur_id, seq_len, seq_size, emb_dim, hist_gru, combine_mechanism):
        hist_attention = self.attention_net(cur_id, hist_gru, seq_len, seq_size, emb_dim)
        logging.info("%s interest_evolving_layer, hist_gru.shape:%s, hist_attention.shape:%s" % (
            filename, str(hist_gru.shape), str(hist_attention.shape)))

        if combine_mechanism == "AUGRU":
            pass
        elif combine_mechanism == "AGRU":
            pass
        elif combine_mechanism == "AIGRU":
            return self.AIGRU(name, seq_size, emb_dim, hist_gru, hist_attention)
        else:
            raise ValueError("%s interest_evolving_layer, combine_mechanism: %s not supported." % (filename, combine_mechanism))

    def attention_net(self, cur_id, hist_gru, seq_len, seq_size, emb_dim):
        # scaled Dot-Product
        hist_attention = tf.matmul(tf.expand_dims(cur_id, 1), hist_gru, transpose_b=True) / (emb_dim ** -0.5)
        # mask
        seq_len = tf.expand_dims(seq_len, 1)
        mask = tf.sequence_mask(seq_len, seq_size)
        padding = tf.ones_like(hist_attention) * (-2 ** 32 + 1)
        hist_attention = tf.where(mask, hist_attention, padding)
        # scale
        hist_attention = tf.nn.softmax(hist_attention)
        return hist_attention

    def AIGRU(self, name, seq_size, emb_dim, hist_gru, hist_attention):
        hist_gru = tf.math.multiply(hist_gru, tf.reshape(hist_attention, (-1, seq_size, 1)))
        final_state = tf.keras.layers.GRU(
            units=emb_dim,
            return_sequences=False,
            go_backwards=True,
            name="%s_AIGRU" % name,
        )(hist_gru)
        return final_state

    def AGRU(self):
        """
        TODO
        Returns:

        """
        return None

    def AUGRU(self):
        """
        TODO
        Returns:

        """
        return None


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
                seq_size=tower.dien_config.seq_size,
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
