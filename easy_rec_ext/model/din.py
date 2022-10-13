# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/23 2:14 下午
# desc: Deep Interest Network

import logging

import tensorflow as tf
from easy_rec_ext.layers import dnn
from easy_rec_ext.core import regularizers
from easy_rec_ext.model.rank_model import RankModel

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


class DINConfig(object):
    def __init__(self, dnn_config: dnn.DNNConfig, need_scale=True, return_target=True, limit_seq_size=None):
        self.dnn_config = dnn_config
        self.need_scale = need_scale
        self.return_target = return_target
        self.limit_seq_size = limit_seq_size

    @staticmethod
    def handle(data):
        dnn_config = dnn.DNNConfig.handle(data["dnn_config"])
        res = DINConfig(dnn_config)
        if "need_scale" in data:
            res.need_scale = data["need_scale"]
        if "return_target" in data:
            res.return_target = data["return_target"]
        if "limit_seq_size" in data:
            res.limit_seq_size = data["limit_seq_size"]
        return res

    def __str__(self):
        return str(self.__dict__)


class DINTower(object):
    def __init__(self, input_group, din_config: DINConfig):
        self.input_group = input_group
        self.din_config = din_config

    @staticmethod
    def handle(data):
        din_config = DINConfig.handle(data["din_config"])
        res = DINTower(data["input_group"], din_config)
        return res

    def __str__(self):
        return str(self.__dict__)


class DINLayer(object):
    def din(self, dnn_config, deep_fea, name, need_scale=True, return_target=True, limit_seq_size=None):
        cur_id, hist_id_col, seq_len = deep_fea["key"], deep_fea["hist_seq_emb"], deep_fea["hist_seq_len"]

        if limit_seq_size and limit_seq_size > 0:
            cur_batch_max_seq_len = tf.shape(hist_id_col)[1]
            hist_id_col = tf.cond(
                tf.constant(limit_seq_size) > cur_batch_max_seq_len,
                lambda: tf.pad(hist_id_col, [[0, 0], [0, limit_seq_size - cur_batch_max_seq_len], [0, 0]],
                               "CONSTANT"),
                lambda: tf.slice(hist_id_col, [0, 0, 0], [-1, limit_seq_size, -1])
            )
            seq_len = tf.where(
                tf.math.less_equal(seq_len, limit_seq_size),
                seq_len,
                limit_seq_size * tf.ones_like(seq_len),
            )

        seq_max_len = tf.shape(hist_id_col)[1]
        emb_dim = hist_id_col.shape[2]

        cur_ids = tf.tile(cur_id, [1, seq_max_len])
        cur_ids = tf.reshape(
            cur_ids,
            tf.shape(hist_id_col)
        )  # (B, seq_max_len, emb_dim)

        din_net = tf.concat(
            [cur_ids, hist_id_col, cur_ids - hist_id_col, cur_ids * hist_id_col],
            axis=-1
        )  # (B, seq_max_len, emb_dim*4)

        assert dnn_config.hidden_units[-1] == 1
        din_net = dnn.DNN(dnn_config, None, name=name + "_attention")(din_net)
        scores = tf.reshape(din_net, [-1, 1, seq_max_len])  # (B, 1, ?)

        # mask
        seq_len = tf.expand_dims(seq_len, 1)
        mask = tf.sequence_mask(seq_len, seq_max_len)
        if need_scale:
            padding = tf.ones_like(scores) * (-2 ** 32 + 1)
        else:
            padding = tf.zeros_like(scores)
        scores = tf.where(mask, scores, padding)  # [B, 1, seq_max_len]

        # Scale
        if need_scale:
            scores = tf.nn.softmax(scores)  # (B, 1, seq_max_len)

        din_output = tf.matmul(scores, hist_id_col)  # [B, 1, emb_dim]
        din_output = tf.reshape(din_output, [-1, emb_dim])  # [B, emb_dim]

        if return_target:
            din_output = tf.concat([din_output, cur_id], axis=1)
        logging.info("din %s, din_output.shape:%s" % (name, str(din_output.shape)))
        return din_output


class DIN(RankModel, DINLayer):
    def __init__(self,
                 model_config,
                 feature_config,
                 features,
                 labels=None,
                 is_training=False):
        super(DIN, self).__init__(model_config, feature_config, features, labels, is_training)

        self._dnn_tower_num = len(self._model_config.dnn_towers) if self._model_config.dnn_towers else 0
        self._dnn_tower_features = []
        for tower_id in range(self._dnn_tower_num):
            tower = self._model_config.dnn_towers[tower_id]
            tower_feature = self.build_input_layer(tower.input_group)
            self._dnn_tower_features.append(tower_feature)

        self._din_tower_num = len(self._model_config.din_towers)
        self._din_tower_features = []
        for tower_id in range(self._din_tower_num):
            tower = self._model_config.din_towers[tower_id]
            tower_feature = self.build_seq_att_input_layer(tower.input_group)
            regularizers.apply_regularization(self._emb_reg, weights_list=[tower_feature["key"]])
            regularizers.apply_regularization(self._emb_reg, weights_list=[tower_feature["hist_seq_emb"]])
            self._din_tower_features.append(tower_feature)

        logging.info("all tower num: {0}".format(self._dnn_tower_num + self._din_tower_num))
        logging.info("dnn tower num: {0}".format(self._dnn_tower_num))
        logging.info("din tower num: {0}".format(self._din_tower_num))

    def build_predict_graph(self):
        tower_fea_arr = []

        for tower_id in range(self._dnn_tower_num):
            tower_fea = self._dnn_tower_features[tower_id]
            tower = self._model_config.dnn_towers[tower_id]
            tower_fea = tf.layers.batch_normalization(
                tower_fea,
                training=self._is_training,
                trainable=True,
                name="%s_fea_bn" % tower.input_group)
            dnn_layer = dnn.DNN(tower.dnn_config, self._l2_reg, "%s_dnn" % tower.input_group,
                                self._is_training)
            tower_fea = dnn_layer(tower_fea)
            tower_fea_arr.append(tower_fea)

        for tower_id in range(self._din_tower_num):
            tower_fea = self._din_tower_features[tower_id]
            tower = self._model_config.din_towers[tower_id]
            tower_fea = self.din(tower.din_config.dnn_config,
                                 tower_fea,
                                 name="%s_din" % tower.input_group,
                                 need_scale=tower.din_config.need_scale,
                                 return_target=tower.din_config.return_target,
                                 limit_seq_size=tower.din_config.limit_seq_size,
                                 )
            tower_fea_arr.append(tower_fea)

        all_fea = tf.concat(tower_fea_arr, axis=1)
        final_dnn_layer = dnn.DNN(self._model_config.final_dnn, self._l2_reg, "final_dnn", self._is_training)
        all_fea = final_dnn_layer(all_fea)
        logging.info("DIN build_predict_graph, all_fea.shape:%s" % (str(all_fea.shape)))

        logits = tf.layers.dense(all_fea, 1, name="logits")
        logits = tf.reshape(logits, (-1,))
        probs = tf.sigmoid(logits, name="probs")

        prediction_dict = dict()
        prediction_dict["probs"] = probs
        self._add_to_prediction_dict(prediction_dict)
        return self._prediction_dict
